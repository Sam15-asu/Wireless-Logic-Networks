import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class ThresholdLayer(nn.Module):
    __constants__ = ['bias']

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_active: int,
        bias: bool = True,
        grad_factor: float = 1.0,
        implementation: str = 'python',
        layer_id: int = None,
        idx: torch.Tensor | None = None,   # optional explicit connectivity
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_active = num_active
        self.grad_factor = grad_factor
        self.layer_id = layer_id
        self.implementation = implementation
        assert implementation in ['python'], implementation

        # Fixed connectivity: (out_dim, num_active)
        if idx is None:
            idx = self.create_indices(out_dim, in_dim, num_active)
        else:
            assert idx.shape == (out_dim, num_active)
            assert idx.dtype == torch.long

        self.register_buffer('idx', idx)

        # Trainable weights only for active connections: (out_dim, num_active)
        self.weight = nn.Parameter(torch.empty(out_dim, num_active))
        nn.init.xavier_uniform_(self.weight)

        if bias:
            self.bias = nn.Parameter(torch.empty(out_dim))
            # bias init similar spirit to your current code (fan_in = num_active)
            bound = 1.0 / (num_active ** 0.5) if num_active > 0 else 0.0
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

        self.num_neurons = out_dim
        self.num_weights = out_dim * num_active  # true parameter count now

    @staticmethod
    def create_indices(out_dim: int, in_dim: int, num_active: int) -> torch.Tensor:
        """
        Build (out_dim, num_active) indices with:
          - exactly num_active inputs per neuron (row)
          - tries to cover all columns at least once
        """
        # Start with random indices per row
        idx = torch.randint(0, in_dim, (out_dim, num_active), dtype=torch.long)

        # Optional: enforce uniqueness within each row (helps stability)
        # If in_dim is large relative to num_active, this almost always succeeds quickly.
        for r in range(out_dim):
            # re-sample until unique (cheap for num_active=6)
            while True:
                row = idx[r]
                if torch.unique(row).numel() == num_active:
                    break
                idx[r] = torch.randint(0, in_dim, (num_active,), dtype=torch.long)

        # Optional: try to cover all input columns at least once
        # (This is a best-effort heuristic; strict coverage requires a more careful construction.)
        used = torch.zeros(in_dim, dtype=torch.bool)
        used[idx.reshape(-1)] = True
        missing = (~used).nonzero(as_tuple=True)[0]

        if missing.numel() > 0:
            # Replace some random positions with missing columns
            flat = idx.view(-1)
            replace_positions = torch.randperm(flat.numel())[:missing.numel()]
            flat[replace_positions] = missing
            idx = flat.view(out_dim, num_active)

        return idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.grad_factor != 1.0:
            x = GradFactor.apply(x, self.grad_factor)

        # x: (B, in_dim)
        # gather -> (B, out_dim, num_active)
        x_sel = x[:, self.idx]

        # weighted sum -> (B, out_dim)
        y = (x_sel * self.weight).sum(dim=-1)

        if self.bias is not None:
            y = y - self.bias
        return y

    def extra_repr(self):
        return f'in_dim={self.in_dim}, out_dim={self.out_dim}, num_active={self.num_active}, bias={self.bias is not None}'

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class ThresholdLayerFC(nn.Module):
    """
    Fully-connected version of ThresholdLayer:
    y = x @ W^T - b   (matches your current forward_python sign)
    """
    __constants__ = ['bias']

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        device: str = 'cuda',
        grad_factor: float = 1.,
        implementation: str = 'python',
        layer_id: int = None,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device
        self.grad_factor = grad_factor
        self.layer_id = layer_id
        self.implementation = implementation
        assert self.implementation in ['python'], (self.implementation)

        self.num_neurons = out_dim
        self.num_weights = out_dim * in_dim

        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_dim))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight)
        if self.bias is not None:
            # standard-ish bias init based on fan_in
            bound = 1.0 / math.sqrt(self.in_dim) if self.in_dim > 0 else 0.0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        if self.grad_factor != 1.:
            x = GradFactor.apply(x, self.grad_factor)  # keep your behavior if GradFactor exists
        if self.implementation == 'python':
            return self.forward_python(x)
        raise ValueError(self.implementation)

    def forward_python(self, x):
        # Fully connected linear transform
        # keep your convention: subtract bias
        return F.linear(x, self.weight, None) - (self.bias if self.bias is not None else 0.0)

    def extra_repr(self):
        return f"in_dim={self.in_dim}, out_dim={self.out_dim}, bias={self.bias is not None}"
    
    

class CustomSigmoid(nn.Module):
    """
    A customized sigmoid activation function that applies a trainable scaling parameter s.
    
    During training, it computes:
        output = sigmoid(s * x)
    
    During evaluation, it computes a hard threshold (Heaviside step):
        output = 1 if s * x > 0, else 0
    """
    def __init__(self, init_scale: float = 1.0, layer_id: int = None):
        """
        Args:
            init_scale (float): Initial value for the trainable scaling parameter s.
        """
        super(CustomSigmoid, self).__init__()
        self.s = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))
        self.layer_id = layer_id
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scaled_x = self.s * x
    
        # Return outputs based on mode.
        if self.training:
            sigmoid_output = torch.sigmoid(scaled_x)
            return sigmoid_output
        
        else:
            y_scaled = (scaled_x >= 0).float()
            return y_scaled
        
    def extra_repr(self) -> str:
        return f'scale={self.s.item()}'
    


class CustomSigmoid2(nn.Module):
    """
    Tanh gate with a monotonically increasing scale 'p' during training.
    - train():  tanh(p * x), then p += step
    - eval():   hard step: 1 if p*x >= 0 else 0
    """
    def __init__(self, init_scale: float = 1.0, step: float = 0.0007):
        super().__init__()
        # keep as a real buffer so it stays on-device and in state_dict
        self.register_buffer('p', torch.tensor(float(init_scale), dtype=torch.float32))
        self.step = float(step)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # update buffer in-place (do not rebind self.p)
            with torch.no_grad():
                self.p.add_(self.step)
            return torch.tanh(self.p * x)
        else:
            return ((self.p * x) >= 0).to(x.dtype)

    def extra_repr(self) -> str:
        return f'scale={self.p.item()} step={self.step}'
 
class TanhForwardSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p):
        return torch.tanh(p * x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class CustomSigmoid3(nn.Module):
    """
    Forward: tanh(p*x)
    Backward: straight-through estimator (identity gradient)
    Eval: hard threshold
    """
    def __init__(self, init_scale: float = 1.0, step: float = 0.0007):
        super().__init__()
        self.register_buffer("p", torch.tensor(float(init_scale), dtype=torch.float32))
        self.step = float(step)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            with torch.no_grad():
                self.p.add_(self.step)
            return TanhForwardSTE.apply(x, self.p)
        else:
            return ((self.p * x) >= 0).to(x.dtype)

    def extra_repr(self) -> str:
        return f"scale={self.p.item()} step={self.step}"

class GroupSum(torch.nn.Module):
    """
    The GroupSum module.
    """
    def __init__(self, k: int, tau: float = 1., device='cuda'):
        """

        :param k: number of intended real valued outputs, e.g., number of classes
        :param tau: the (softmax) temperature tau. The summed outputs are divided by tau.
        :param device:
        """
        super().__init__()
        self.k = k
        self.tau = tau
        self.device = device

    def forward(self, x):
        assert x.shape[-1] % self.k == 0, (x.shape, self.k)
        
        out = x.reshape(*x.shape[:-1], self.k, x.shape[-1] // self.k).sum(-1) / self.tau
        
        return out
    print("----------------------------------------------------------------------------")
        
    def extra_repr(self):
        return 'k={}, tau={}'.format(self.k, self.tau)
    


class GradFactor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, f):
        ctx.f = f
        return x

    @staticmethod
    def backward(ctx, grad_y):
        return grad_y * ctx.f, None
    

class StepSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        # Surrogate gradient: nonzero only near 0
        # This is like hardtanh' window; adjust width to taste.
        width = 1.0
        mask = (x.abs() <= width).to(x.dtype)
        return grad_out * mask


class StepGateSTE(nn.Module):
    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.register_buffer("p", torch.tensor(float(scale), dtype=torch.float32))

    def forward(self, x):
        return StepSTE.apply(self.p * x)

    def extra_repr(self):
        return f"scale={self.p.item()}"
    
class ClippedStepSTE(torch.autograd.Function):
    """
    Forward: hard step (0/1)
    Backward: identity gradient inside |x| <= w, else 0
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, w: float):
        ctx.save_for_backward(x)
        ctx.w = float(w)
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (x,) = ctx.saved_tensors
        w = ctx.w
        mask = (x.abs() <= w).to(grad_out.dtype)
        grad_x = grad_out * mask
        return grad_x, None  # None for w (not trainable)


class StepGateClippedSTE(nn.Module):
    """
    Drop-in gate module:
      - train/eval forward: hard step on (p * x)
      - backward (training): clipped STE with window w on (p * x)
    """
    def __init__(self, scale: float = 1.0, w: float = 1.0):
        super().__init__()
        self.register_buffer("p", torch.tensor(float(scale), dtype=torch.float32))
        self.w = float(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ClippedStepSTE.apply(self.p * x, self.w)

    def extra_repr(self) -> str:
        return f"scale={self.p.item()} w={self.w}"