import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import math


# --------------------------
# Clipped STE (0/1 forward, clipped identity grad)
# --------------------------
class ClippedStepSTE(torch.autograd.Function):
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
        return grad_out * mask, None


class StepGateClippedSTE(nn.Module):
    def __init__(self, scale: float = 1.0, w: float = 1.0):
        super().__init__()
        self.register_buffer("p", torch.tensor(float(scale), dtype=torch.float32))
        self.w = float(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ClippedStepSTE.apply(self.p * x, self.w)

    def extra_repr(self) -> str:
        return f"scale={self.p.item()} w={self.w}"



class SparseThresholdLinear(nn.Module):
    """
    Sparse wired *pre-activation* thresholded linear block.

    Computes: z = sum_i w_i * x_i - theta

    No nonlinearity inside. You can attach an external activation (STE, sigmoid, etc.)
    in your model definition.

    Input:  x shape (B, in_dim) or (B, in_dim, L)
    Output: z shape (B, out_dim) or (B, out_dim, L)

    Wiring:
      - fixed idx of shape (out_dim, fan_in) picking fan_in inputs per output unit
      - learnable w of shape (out_dim, fan_in)
      - learnable theta of shape (out_dim,)
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        fan_in: int = 8,
        weight_init: str = "xavier_uniform",
        theta_init: str = "mean_abs_w",
        layer_id: int = None,
        idx: torch.Tensor | None = None,
    ):
        super().__init__()
        assert 1 <= fan_in <= in_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.fan_in = fan_in
        self.layer_id = layer_id

        # ---- fixed sparse wiring ----
        if idx is None:
            idx = self._make_connections(in_dim, out_dim, fan_in).long()
        else:
            assert idx.shape == (out_dim, fan_in), f"idx must be (out_dim, fan_in)=({out_dim},{fan_in})"
            assert idx.dtype in (torch.int64, torch.int32), "idx must be integer type"
            assert idx.min().item() >= 0 and idx.max().item() < in_dim, "idx values must be in [0, in_dim)"
        self.register_buffer("idx", idx.long())  # (out_dim, fan_in)

        # ---- learnable params ----
        self.w = nn.Parameter(torch.empty(out_dim, fan_in))
        self.theta = nn.Parameter(torch.zeros(out_dim))

        self._init_weights(self.w, scheme=weight_init)
        self._init_theta(theta_init)

    @staticmethod
    def _make_connections(in_dim: int, out_dim: int, fan_in: int) -> torch.Tensor:
        rows = []
        for _ in range(out_dim):
            perm = torch.randperm(in_dim)  # no replacement within a unit
            rows.append(perm[:fan_in])
        return torch.stack(rows, dim=0)    # (out_dim, fan_in)

    def _init_weights(self, w: torch.Tensor, scheme: str):
        if scheme == "kaiming_uniform":
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        elif scheme == "xavier_uniform":
            nn.init.xavier_uniform_(w)
        elif scheme == "normal_small":
            nn.init.normal_(w, mean=0.0, std=0.1)
        else:
            raise ValueError(f"Unknown weight_init: {scheme}")

    def _init_theta(self, mode: str):
        with torch.no_grad():
            if mode == "zero":
                self.theta.zero_()
            elif mode == "mean_abs_w":
                t = self.w.abs().mean(dim=1) * (self.fan_in / 2.0)
                self.theta.copy_(t)
            elif mode == "median_abs_w":
                t = self.w.abs().median(dim=1).values * (self.fan_in / 2.0)
                self.theta.copy_(t)
            else:
                raise ValueError(f"Unknown theta_init: {mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, in_dim) or (B, in_dim, L)
        returns z: (B, out_dim) or (B, out_dim, L)
        """
        assert x.dim() in (2, 3), "x must be (B, in_dim) or (B, in_dim, L)"
        B = x.size(0)
        extra = x.shape[2:]  # () or (L,)

        # Gather wired inputs: (B, out_dim, fan_in, *extra)
        x_sel = x[:, self.idx.reshape(-1), ...]  # (B, out_dim*fan_in, *extra)
        x_sel = x_sel.contiguous().view(B, self.out_dim, self.fan_in, *extra)

        # Weighted sum
        w = self.w.contiguous().view(1, self.out_dim, self.fan_in, *([1] * len(extra)))
        sum_ = (x_sel * w).sum(dim=2)            # (B, out_dim, *extra)

        # Threshold (pre-activation)
        theta = self.theta.view(1, self.out_dim, *([1] * len(extra)))
        z = sum_ - theta
        return z


class SparseChannelLockedConv(nn.Module):
    """
    "Logic conv" WITHOUT activation: unfold -> sparse threshold linear -> reshape.

    Key constraint (same as your current code):
      Each output channel j is assigned exactly ONE input channel c_j (fixed),
      and its sparse taps are restricted to that channel's k*k patch only.

    Output is pre-activation z. Apply STE (or any activation) outside this module.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        *,
        fan_in: int = 6,
        weight_init: str = "kaiming_uniform",
        theta_init: str = "mean_abs_w",
    ):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.stride = int(stride)
        self.padding = int(padding)

        k2 = self.kernel_size ** 2
        in_dim = k2 * self.in_channels

        # fan_in cannot exceed k2 because each output sees only one channel's k*k patch
        fan_in = min(int(fan_in), k2)

        # Fixed random assignment: each output channel j picks one input channel c_j.
        # When enough output channels are available, guarantee each input channel is
        # selected at least once across the layer.
        chan_idx = self._make_channel_assignments(self.in_channels, self.out_channels)
        self.register_buffer("chan_idx", chan_idx)  # (Cout,)

        # Random taps inside the local k*k patch (indices in [0..k2-1])
        idx_local = SparseThresholdLinear._make_connections(k2, self.out_channels, fan_in).long()  # (Cout, fan_in)

        # Convert local indices to global unfolded indices: global = c_j*k2 + local
        idx_global = self.chan_idx[:, None] * k2 + idx_local  # (Cout, fan_in)

        # Pre-activation gatebank (no sigmoid/step here)
        self.gatebank = SparseThresholdLinear(
            in_dim=in_dim,
            out_dim=self.out_channels,
            fan_in=fan_in,
            weight_init=weight_init,
            theta_init=theta_init,
            idx=idx_global,
        )

    @staticmethod
    def _make_channel_assignments(in_channels: int, out_channels: int) -> torch.Tensor:
        """
        Build a length-`out_channels` tensor assigning one input channel to each
        output channel.

        If out_channels >= in_channels, every input channel appears at least once.
        Otherwise, exact coverage is impossible, so assignments are still random but
        only cover a subset of input channels.
        """
        if out_channels <= 0:
            return torch.empty(0, dtype=torch.long)

        if out_channels >= in_channels:
            # Start with one copy of every input channel, then fill the remaining
            # kernels with random channel choices, and shuffle the final assignment.
            base = torch.arange(in_channels, dtype=torch.long)
            extra_count = out_channels - in_channels
            if extra_count > 0:
                extra = torch.randint(low=0, high=in_channels, size=(extra_count,), dtype=torch.long)
                chan_idx = torch.cat([base, extra], dim=0)
            else:
                chan_idx = base
            return chan_idx[torch.randperm(out_channels)]

        # Not enough output channels to cover every input channel; choose a random
        # subset without replacement so assignments are distinct when possible.
        return torch.randperm(in_channels, dtype=torch.long)[:out_channels]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, Cin, H, W)
        returns z: (B, Cout, Hout, Wout)   (pre-activation)
        """
        B, _, H, W = x.shape
        x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode="constant", value=0)

        Hout = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        Wout = (W + 2 * self.padding - self.kernel_size) // self.stride + 1

        patches = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride)  # (B, k*k*Cin, L)
        z = self.gatebank(patches)                                                # (B, Cout, L) pre-activation
        z = einops.rearrange(z, "b c (h w) -> b c h w", h=Hout, w=Wout)
        return z
    

# --------------------------
# Two-level threshold bank
# Level1: 4 gates, fan_in=2, STE -> 4 bits
# Level2: 1 gate, fan_in=4      -> z2 (pre-activation)
# --------------------------
class TwoLevelThresholdGateBank(nn.Module):
    """
    Input x: (B, in_dim, L)
    Output z2: (B, out_dim, L)  (pre-activation for level2)

    IMPORTANT GUARANTEE (when idx1 is None):
      For each output unit j, level-1 chooses 8 DISTINCT inputs (no overlap across the 4 two-input gates).
      So the 4 gates consume 8 unique indices total.
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        num_level1: int = 4,
        fan_in1: int = 2,
        fan_in2: int = 4,
        idx1: torch.Tensor | None = None,  # (out_dim, 4, 2)
        ste_scale: float = 1.0,
        ste_window: float = 1.0,
        weight_init: str = "xavier_uniform",
        theta_init: str = "mean_abs_w",
    ):
        super().__init__()
        assert num_level1 == 4, "Spec: 4 two-input gates at level 1"
        assert fan_in1 == 2, "Spec: fan-in=2 at level 1"
        assert fan_in2 == 4, "Spec: fan-in=4 at level 2"
        assert in_dim >= 8, "Need at least 8 distinct inputs to avoid overlap across level-1 gates."

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_level1 = num_level1
        self.fan_in1 = fan_in1
        self.fan_in2 = fan_in2

        # ---- wiring for level1 ----
        if idx1 is None:
            idx1 = self._make_level1_connections_disjoint(in_dim, out_dim)
        else:
            assert idx1.shape == (out_dim, num_level1, fan_in1)
            assert idx1.dtype in (torch.int64, torch.int32)
            assert idx1.min().item() >= 0 and idx1.max().item() < in_dim
            # ensure disjointness per output (optional but recommended)
            flat = idx1.view(out_dim, -1)  # (out, 8)
            # this asserts all 8 are unique for each output
            if not torch.all(torch.tensor([torch.unique(flat[j]).numel() == 8 for j in range(out_dim)])):
                raise ValueError("idx1 must use 8 distinct indices per output (no overlap across 4 two-input gates).")
        self.register_buffer("idx1", idx1.long())

        # ---- parameters: level1 ----
        self.w1 = nn.Parameter(torch.empty(out_dim, num_level1, fan_in1))  # (out,4,2)
        self.theta1 = nn.Parameter(torch.zeros(out_dim, num_level1))       # (out,4)

        # ---- STE between levels ----
        self.ste = StepGateClippedSTE(scale=ste_scale, w=ste_window)

        # ---- parameters: level2 ----
        self.w2 = nn.Parameter(torch.empty(out_dim, fan_in2))              # (out,4)
        self.theta2 = nn.Parameter(torch.zeros(out_dim))                   # (out,)

        # init
        self._init_weights(self.w1, scheme=weight_init)
        self._init_weights(self.w2, scheme=weight_init)
        self._init_theta1(theta_init)
        self._init_theta2(theta_init)

    @staticmethod
    def _make_level1_connections_disjoint(in_dim: int, out_dim: int) -> torch.Tensor:
        """
        Returns idx1 of shape (out_dim, 4, 2) such that for each output j:
          - 8 distinct indices are used across the 4 gates (no overlap across gates)
        """
        idx = torch.empty(out_dim, 4, 2, dtype=torch.long)
        for j in range(out_dim):
            perm = torch.randperm(in_dim)
            chosen8 = perm[:8]          # 8 unique inputs
            idx[j] = chosen8.view(4, 2) # partition into 4 disjoint pairs
        return idx

    def _init_weights(self, w: torch.Tensor, scheme: str):
        if scheme == "kaiming_uniform":
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        elif scheme == "xavier_uniform":
            nn.init.xavier_uniform_(w)
        elif scheme == "normal_small":
            nn.init.normal_(w, mean=0.0, std=0.1)
        else:
            raise ValueError(f"Unknown weight_init: {scheme}")

    def _init_theta1(self, mode: str):
        with torch.no_grad():
            if mode == "zero":
                self.theta1.zero_()
            elif mode == "mean_abs_w":
                t = self.w1.abs().mean(dim=2) * (self.fan_in1 / 2.0)  # (out,4)
                self.theta1.copy_(t)
            elif mode == "median_abs_w":
                t = self.w1.abs().median(dim=2).values * (self.fan_in1 / 2.0)
                self.theta1.copy_(t)
            else:
                raise ValueError(f"Unknown theta_init: {mode}")

    def _init_theta2(self, mode: str):
        with torch.no_grad():
            if mode == "zero":
                self.theta2.zero_()
            elif mode == "mean_abs_w":
                t = self.w2.abs().mean(dim=1) * (self.fan_in2 / 2.0)  # (out,)
                self.theta2.copy_(t)
            elif mode == "median_abs_w":
                t = self.w2.abs().median(dim=1).values * (self.fan_in2 / 2.0)
                self.theta2.copy_(t)
            else:
                raise ValueError(f"Unknown theta_init: {mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, in_dim, L)
        returns z2: (B, out_dim, L)
        """
        assert x.dim() == 3, "Expected x = (B, in_dim, L)"
        B, _, L = x.shape

        # Gather level1 inputs: idx1 is (out,4,2)
        idx_flat = self.idx1.reshape(-1)                    # (out*4*2,)
        x_sel = x[:, idx_flat, :]                           # (B, out*4*2, L)
        x_sel = x_sel.view(B, self.out_dim, 4, 2, L)        # (B,out,4,2,L)

        # Level1 preactivation
        w1 = self.w1.view(1, self.out_dim, 4, 2, 1)
        sum1 = (x_sel * w1).sum(dim=3)                      # (B,out,4,L)
        theta1 = self.theta1.view(1, self.out_dim, 4, 1)
        z1 = sum1 - theta1                                  # (B,out,4,L)

        # STE -> 4 binary features
        h1 = self.ste(z1)                                   # (B,out,4,L) forward in {0,1}

        # Level2 preactivation
        w2 = self.w2.view(1, self.out_dim, 4, 1)
        sum2 = (h1 * w2).sum(dim=2)                         # (B,out,L)
        theta2 = self.theta2.view(1, self.out_dim, 1)
        z2 = sum2 - theta2
        return z2


# --------------------------
# Two-level channel-locked conv with DISJOINT 8-of-9 pixel selection
# --------------------------
class TwoLevelChannelLockedConv(nn.Module):
    """
    unfold -> TwoLevelThresholdGateBank -> reshape

    IMPORTANT GUARANTEE:
      For each output channel j (kernel), we pick ONE input channel c_j.
      Within that channel's k*k patch, we pick 8 DISTINCT pixels (out of 9 for k=3),
      split into 4 disjoint pairs for the 4 two-input gates.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        *,
        ste_scale: float = 1.0,
        ste_window: float = 1.0,
        weight_init: str = "xavier_uniform",
        theta_init: str = "mean_abs_w",
    ):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.stride = int(stride)
        self.padding = int(padding)

        k2 = self.kernel_size * self.kernel_size
        in_dim = k2 * self.in_channels

        # fixed channel assignment per output
        chan_idx = torch.randint(0, self.in_channels, (self.out_channels,), dtype=torch.long)
        self.register_buffer("chan_idx", chan_idx)  # (out,)

        # Build idx1_local with DISJOINT pairs (8 distinct pixels) per output
        assert k2 >= 8, "Need at least 8 pixels in the local patch to pick 8 distinct ones."
        idx1_local = torch.empty(self.out_channels, 4, 2, dtype=torch.long)
        for j in range(self.out_channels):
            perm = torch.randperm(k2)      # permutation of [0..k2-1]
            chosen8 = perm[:8]             # 8 distinct pixels from the k*k patch
            idx1_local[j] = chosen8.view(4, 2)

        # map local -> global indices into unfolded vector
        idx1_global = self.chan_idx[:, None, None] * k2 + idx1_local  # (out,4,2)

        self.gatebank = TwoLevelThresholdGateBank(
            in_dim=in_dim,
            out_dim=self.out_channels,
            idx1=idx1_global,
            ste_scale=ste_scale,
            ste_window=ste_window,
            weight_init=weight_init,
            theta_init=theta_init,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, Cin, H, W)
        returns z2: (B, Cout, Hout, Wout)
        """
        B, _, H, W = x.shape
        x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), value=0)

        Hout = (H + 2*self.padding - self.kernel_size) // self.stride + 1
        Wout = (W + 2*self.padding - self.kernel_size) // self.stride + 1

        patches = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride)  # (B, k2*Cin, L)
        z2 = self.gatebank(patches)                                               # (B, Cout, L)
        z2 = einops.rearrange(z2, "b c (h w) -> b c h w", h=Hout, w=Wout)
        return z2

# --------------------------
# Two-level threshold bank:
# Level 1:
#   - one 4-input threshold gate
#   - one 5-input threshold gate
#   - both use disjoint raw inputs
#   - both go through STE
#
# Level 2:
#   - one 2-input threshold gate over the two level-1 outputs
#   - also goes through STE
#
# Input:  x  -> (B, in_dim, L)
# Output: y2 -> (B, out_dim, L)   binary forward, STE backward
# --------------------------
class TwoGate45Then2ThresholdBank(nn.Module):
    """
    For each output unit j:
      Level 1:
        z1a = sum_{i=1..4} w1a_i * x[idx1a_i] - theta1a
        z1b = sum_{i=1..5} w1b_i * x[idx1b_i] - theta1b
        h1a = STE(z1a)
        h1b = STE(z1b)

      Level 2:
        z2 = w2_0 * h1a + w2_1 * h1b - theta2
        y2 = STE(z2)

    idx1a and idx1b are disjoint per output unit.
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        idx1a: torch.Tensor | None = None,   # (out_dim, 4)
        idx1b: torch.Tensor | None = None,   # (out_dim, 5)
        ste_scale: float = 1.0,
        ste_window: float = 1.0,
        weight_init: str = "xavier_uniform",
        theta_init: str = "mean_abs_w",
    ):
        super().__init__()
        assert in_dim >= 9, "Need at least 9 distinct inputs for disjoint 4-input and 5-input gates."

        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)

        # ---- level-1 wiring ----
        if idx1a is None or idx1b is None:
            idx1a_gen, idx1b_gen = self._make_disjoint_level1_connections(self.in_dim, self.out_dim)
            if idx1a is None:
                idx1a = idx1a_gen
            if idx1b is None:
                idx1b = idx1b_gen

        assert idx1a.shape == (self.out_dim, 4), f"idx1a must be ({self.out_dim}, 4)"
        assert idx1b.shape == (self.out_dim, 5), f"idx1b must be ({self.out_dim}, 5)"
        assert idx1a.dtype in (torch.int32, torch.int64), "idx1a must be integer type"
        assert idx1b.dtype in (torch.int32, torch.int64), "idx1b must be integer type"
        assert idx1a.min().item() >= 0 and idx1a.max().item() < self.in_dim
        assert idx1b.min().item() >= 0 and idx1b.max().item() < self.in_dim

        for j in range(self.out_dim):
            a = idx1a[j]
            b = idx1b[j]
            ua = torch.unique(a)
            ub = torch.unique(b)
            uab = torch.unique(torch.cat([a, b], dim=0))
            if ua.numel() != 4:
                raise ValueError(f"idx1a[{j}] must contain 4 distinct indices.")
            if ub.numel() != 5:
                raise ValueError(f"idx1b[{j}] must contain 5 distinct indices.")
            if uab.numel() != 9:
                raise ValueError(f"idx1a[{j}] and idx1b[{j}] must be disjoint and total 9 distinct indices.")

        self.register_buffer("idx1a", idx1a.long())  # (out,4)
        self.register_buffer("idx1b", idx1b.long())  # (out,5)

        # ---- level-1 params ----
        self.w1a = nn.Parameter(torch.empty(self.out_dim, 4))  # (out,4)
        self.theta1a = nn.Parameter(torch.zeros(self.out_dim)) # (out,)
        self.w1b = nn.Parameter(torch.empty(self.out_dim, 5))  # (out,5)
        self.theta1b = nn.Parameter(torch.zeros(self.out_dim)) # (out,)

        # ---- STE at both levels ----
        self.ste = StepGateClippedSTE(scale=ste_scale, w=ste_window)

        # ---- level-2 params ----
        self.w2 = nn.Parameter(torch.empty(self.out_dim, 2))   # (out,2)
        self.theta2 = nn.Parameter(torch.zeros(self.out_dim))  # (out,)

        self._init_weights(self.w1a, scheme=weight_init)
        self._init_weights(self.w1b, scheme=weight_init)
        self._init_weights(self.w2, scheme=weight_init)

        self._init_theta1(theta_init)
        self._init_theta2(theta_init)

    @staticmethod
    def _make_disjoint_level1_connections(in_dim: int, out_dim: int):
        idx1a = torch.empty(out_dim, 4, dtype=torch.long)
        idx1b = torch.empty(out_dim, 5, dtype=torch.long)
        for j in range(out_dim):
            perm = torch.randperm(in_dim)
            chosen9 = perm[:9]
            idx1a[j] = chosen9[:4]
            idx1b[j] = chosen9[4:]
        return idx1a, idx1b

    def _init_weights(self, w: torch.Tensor, scheme: str):
        if scheme == "kaiming_uniform":
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        elif scheme == "xavier_uniform":
            nn.init.xavier_uniform_(w)
        elif scheme == "normal_small":
            nn.init.normal_(w, mean=0.0, std=0.1)
        else:
            raise ValueError(f"Unknown weight_init: {scheme}")

    def _init_theta1(self, mode: str):
        with torch.no_grad():
            if mode == "zero":
                self.theta1a.zero_()
                self.theta1b.zero_()
            elif mode == "mean_abs_w":
                self.theta1a.copy_(self.w1a.abs().mean(dim=1) * (4.0 / 2.0))
                self.theta1b.copy_(self.w1b.abs().mean(dim=1) * (5.0 / 2.0))
            elif mode == "median_abs_w":
                self.theta1a.copy_(self.w1a.abs().median(dim=1).values * (4.0 / 2.0))
                self.theta1b.copy_(self.w1b.abs().median(dim=1).values * (5.0 / 2.0))
            else:
                raise ValueError(f"Unknown theta_init: {mode}")

    def _init_theta2(self, mode: str):
        with torch.no_grad():
            if mode == "zero":
                self.theta2.zero_()
            elif mode == "mean_abs_w":
                self.theta2.copy_(self.w2.abs().mean(dim=1) * (2.0 / 2.0))
            elif mode == "median_abs_w":
                self.theta2.copy_(self.w2.abs().median(dim=1).values * (2.0 / 2.0))
            else:
                raise ValueError(f"Unknown theta_init: {mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, in_dim, L)
        returns y2: (B, out_dim, L)  binary forward due to STE
        """
        assert x.dim() == 3, "Expected x = (B, in_dim, L)"
        B, _, L = x.shape

        # ---- gather 4-input gate inputs ----
        idx1a_flat = self.idx1a.reshape(-1)                  # (out*4,)
        x1a = x[:, idx1a_flat, :]                            # (B, out*4, L)
        x1a = x1a.view(B, self.out_dim, 4, L)                # (B, out, 4, L)

        # ---- gather 5-input gate inputs ----
        idx1b_flat = self.idx1b.reshape(-1)                  # (out*5,)
        x1b = x[:, idx1b_flat, :]                            # (B, out*5, L)
        x1b = x1b.view(B, self.out_dim, 5, L)                # (B, out, 5, L)

        # ---- level 1: 4-input gate ----
        w1a = self.w1a.view(1, self.out_dim, 4, 1)
        z1a = (x1a * w1a).sum(dim=2) - self.theta1a.view(1, self.out_dim, 1)  # (B, out, L)
        h1a = self.ste(z1a)                                                      # (B, out, L)

        # ---- level 1: 5-input gate ----
        w1b = self.w1b.view(1, self.out_dim, 5, 1)
        z1b = (x1b * w1b).sum(dim=2) - self.theta1b.view(1, self.out_dim, 1)  # (B, out, L)
        h1b = self.ste(z1b)                                                      # (B, out, L)

        # ---- level 2: 2-input gate ----
        h12 = torch.stack([h1a, h1b], dim=2)                                    # (B, out, 2, L)
        w2 = self.w2.view(1, self.out_dim, 2, 1)
        z2 = (h12 * w2).sum(dim=2) - self.theta2.view(1, self.out_dim, 1)      # (B, out, L)
        y2 = self.ste(z2)                                                        # (B, out, L)

        return y2


# --------------------------
# Channel-locked convolution wrapper:
#   unfold -> TwoGate45Then2ThresholdBank -> reshape
#
# For each output channel j:
#   - choose exactly one input channel c_j
#   - choose 9 distinct pixels from that channel's k*k patch
#   - first 4 go to one 4-input threshold gate
#   - remaining 5 go to one 5-input threshold gate
#   - both outputs go through STE
#   - then a 2-input threshold gate -> STE
# --------------------------
class TwoGate45Then2ChannelLockedConv(nn.Module):
    """
    x: (B, Cin, H, W)
    y: (B, Cout, Hout, Wout)   binary forward due to STE

    Important:
    - requires kernel_size^2 >= 9
    - for kernel_size=3, each output channel uses all 9 pixels of one chosen input channel
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        *,
        ste_scale: float = 1.0,
        ste_window: float = 1.0,
        weight_init: str = "xavier_uniform",
        theta_init: str = "mean_abs_w",
    ):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.stride = int(stride)
        self.padding = int(padding)

        k2 = self.kernel_size * self.kernel_size
        in_dim = k2 * self.in_channels

        assert k2 >= 9, "Need kernel_size^2 >= 9 to pick disjoint 4-input and 5-input groups."

        # fixed random assignment: each output channel picks exactly one input channel
        chan_idx = torch.randint(0, self.in_channels, (self.out_channels,), dtype=torch.long)
        self.register_buffer("chan_idx", chan_idx)  # (out,)

        # build local disjoint 4+5 indices inside one channel patch
        idx1a_local = torch.empty(self.out_channels, 4, dtype=torch.long)
        idx1b_local = torch.empty(self.out_channels, 5, dtype=torch.long)

        for j in range(self.out_channels):
            perm = torch.randperm(k2)
            chosen9 = perm[:9]
            idx1a_local[j] = chosen9[:4]
            idx1b_local[j] = chosen9[4:]

        # map local patch indices -> global unfolded indices
        idx1a_global = self.chan_idx[:, None] * k2 + idx1a_local   # (out,4)
        idx1b_global = self.chan_idx[:, None] * k2 + idx1b_local   # (out,5)

        self.gatebank = TwoGate45Then2ThresholdBank(
            in_dim=in_dim,
            out_dim=self.out_channels,
            idx1a=idx1a_global,
            idx1b=idx1b_global,
            ste_scale=ste_scale,
            ste_window=ste_window,
            weight_init=weight_init,
            theta_init=theta_init,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, Cin, H, W)
        returns y: (B, Cout, Hout, Wout)
        """
        B, _, H, W = x.shape

        x = F.pad(
            x,
            (self.padding, self.padding, self.padding, self.padding),
            mode="constant",
            value=0,
        )

        Hout = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        Wout = (W + 2 * self.padding - self.kernel_size) // self.stride + 1

        patches = F.unfold(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
        )  # (B, k2*Cin, L)

        y = self.gatebank(patches)  # (B, Cout, L)
        y = einops.rearrange(y, "b c (h w) -> b c h w", h=Hout, w=Wout)
        return y

class WeightedThresholdGate(nn.Module):
    """
    Weighted-sum threshold gate layer (perceptron-style).

    Input:  x shape (B, in_dim) or (B, in_dim, L)
    Output: y shape (B, out_dim) or (B, out_dim, L)

    Supports optional fixed sparse wiring via `idx` of shape (out_dim, fan_in).
    """
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 fan_in: int = 6,
                 weight_init: str = "xavier_uniform",
                 theta_init: str = "mean_abs_w",
                 s_init: float = 1.0,
                 layer_id: int = None,
                 idx: torch.Tensor = None):
        super().__init__()
        assert 1 <= fan_in
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.fan_in = fan_in
        self.layer_id = layer_id

        # --- fixed sparse wiring ---
        if idx is None:
            assert fan_in <= in_dim
            idx = self._make_connections(in_dim, out_dim, fan_in).long()
        else:
            assert idx.shape == (out_dim, fan_in), f"idx must be (out_dim, fan_in) = ({out_dim},{fan_in})"
            assert idx.dtype in (torch.int64, torch.int32), "idx must be integer type"
            # sanity bounds
            assert idx.min().item() >= 0 and idx.max().item() < in_dim, "idx values must be in [0, in_dim)"
        self.register_buffer("idx", idx.long())  # (out_dim, fan_in)

        # --- learnable params ---
        self.w      = nn.Parameter(torch.empty(out_dim, fan_in))
        self.theta  = nn.Parameter(torch.zeros(out_dim))
        self.s_raw  = nn.Parameter(torch.full((out_dim,), float(s_init)))  # unconstrained

        self._init_weights(self.w, scheme=weight_init)
        self._init_theta(theta_init)

    @staticmethod
    def _make_connections(in_dim: int, out_dim: int, fan_in: int) -> torch.Tensor:
        rows = []
        for _ in range(out_dim):
            perm = torch.randperm(in_dim)        # no replacement within a unit
            rows.append(perm[:fan_in])
        return torch.stack(rows, dim=0)          # (out_dim, fan_in)

    def _init_weights(self, w: torch.Tensor, scheme: str):
        if scheme == "kaiming_uniform":
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        elif scheme == "xavier_uniform":
            nn.init.xavier_uniform_(w)
        elif scheme == "normal_small":
            nn.init.normal_(w, mean=0.0, std=0.1)
        else:
            raise ValueError(f"Unknown weight_init: {scheme}")

    def _init_theta(self, mode: str):
        with torch.no_grad():
            if mode == "zero":
                self.theta.zero_()
            elif mode == "mean_abs_w":
                t = self.w.abs().mean(dim=1) * (self.fan_in / 2.0)
                self.theta.copy_(t)
            elif mode == "median_abs_w":
                t = self.w.abs().median(dim=1).values * (self.fan_in / 2.0)
                self.theta.copy_(t)
            else:
                raise ValueError(f"Unknown theta_init: {mode}")

    def _slope(self) -> torch.Tensor:
        return F.softplus(self.s_raw) + 1e-6  # (out_dim,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, in_dim) or (B, in_dim, L)
        returns: (B, out_dim) or (B, out_dim, L)
        """
        assert x.dim() in (2, 3), "x must be (B, in_dim) or (B, in_dim, L)"
        B = x.size(0)
        extra = x.shape[2:]  # () or (L,)

        # Gather wired inputs: (B, out_dim, fan_in, *extra)
        x_sel = x[:, self.idx.reshape(-1), ...]                 # (B, out_dim*fan_in, *extra)
        x_sel = x_sel.contiguous().view(B, self.out_dim, self.fan_in, *extra)

        # Weighted sum
        w = self.w.contiguous().view(1, self.out_dim, self.fan_in, *([1] * len(extra)))
        sum_ = (x_sel * w).sum(dim=2)                           # (B, out_dim, *extra)

        # Threshold
        theta = self.theta.view(1, self.out_dim, *([1] * len(extra)))
        z = sum_ - theta                                        # (B, out_dim, *extra)

        # Per-output slopes
        s = self._slope().view(1, self.out_dim, *([1] * len(extra)))

        if self.training:
            return torch.sigmoid(s * z)
        else:
            return (z >= 0).to(x.dtype)


class Conv(nn.Module):
    """
    "Logic conv": uses unfold to get (B, k*k*Cin, L), then applies a gatebank.

    MODIFICATION:
      Each output kernel j is randomly assigned exactly ONE input channel c_j.
      Its sparse idx taps are constrained to that channel's k*k block only.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 *, fan_in=6, theta_init="mean_abs_w", s_init=1, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding

        k2 = kernel_size ** 2
        in_dim = k2 * in_channels

        # fan_in cannot exceed k2 because each kernel only sees 1 channel's k*k patch
        fan_in = min(fan_in, k2)

        # Random per-output-channel assignment to one input channel (fixed after init)
        chan_idx = torch.randint(low=0, high=in_channels, size=(out_channels,), dtype=torch.long)
        self.register_buffer("chan_idx", chan_idx)  # (Cout,)

        # Random sparse taps inside the k*k patch (local indices 0..k2-1), per output kernel
        idx_local = WeightedThresholdGate._make_connections(k2, out_channels, fan_in).long()  # (Cout, fan_in)

        # Convert local indices to global indices into the unfolded vector (channel blocks of size k2)
        # global_index = c_j*k2 + local_index
        idx_global = self.chan_idx[:, None] * k2 + idx_local  # (Cout, fan_in)

        # Gatebank uses full unfolded in_dim, but is wired to one channel block per output kernel
        self.gatebank = WeightedThresholdGate(
            in_dim=in_dim,
            out_dim=out_channels,
            fan_in=fan_in,
            weight_init="kaiming_uniform",
            theta_init=theta_init,
            s_init=s_init,
            idx=idx_global
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='constant', value=0)

        Hout = (H + 2*self.padding - self.kernel_size) // self.stride + 1
        Wout = (W + 2*self.padding - self.kernel_size) // self.stride + 1

        patches = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride)  # (B, k*k*Cin, L)
        y = self.gatebank(patches)                                               # (B, Cout, L)
        y = einops.rearrange(y, 'b c (h w) -> b c h w', h=Hout, w=Wout)
        return y

class BranchFusion(torch.nn.Module):
    def __init__(self, branch_a, branch_b):
        super().__init__()
        self.branch_a = branch_a
        self.branch_b = branch_b

    def forward(self, x):
        # x = (x_a, x_b)
        x_a, x_b = x
        return torch.cat([self.branch_a(x_a), self.branch_b(x_b)], dim=1)


class SparseDoubleChannelLockedConv(nn.Module):
    """
    Logic conv WITHOUT activation:

    - Each output channel selects EXACTLY 2 input channels
    - Each selected channel contributes only k×k sparse taps
    - unfold -> sparse linear -> reshape

    Output: pre-activation z
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        *,
        fan_in_per_channel: int = 3,   # total fan_in = 2 * fan_in_per_channel
        weight_init: str = "kaiming_uniform",
        theta_init: str = "mean_abs_w",
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding

        k2 = kernel_size * kernel_size
        in_dim = k2 * in_channels

        fan_in_per_channel = min(fan_in_per_channel, k2)

        # -----------------------------
        # 1) assign EXACTLY 2 channels per output
        # -----------------------------
        chan_pairs = self._make_channel_pairs(in_channels, out_channels)
        self.register_buffer("chan_pairs", chan_pairs)  # (Cout, 2)

        # -----------------------------
        # 2) sparse spatial taps per channel
        # -----------------------------
        idx_local = SparseThresholdLinear._make_connections(
            k2,
            out_channels * 2,  # two channels per output
            fan_in_per_channel
        ).long()  # (Cout*2, fan_in_per_channel)

        idx_local = idx_local.view(out_channels, 2, fan_in_per_channel)

        # -----------------------------
        # 3) convert to global indices
        # -----------------------------
        idx_global = self.chan_pairs[:, :, None] * k2 + idx_local
        # shape: (Cout, 2, fan_in_per_channel)

        self.gatebank = SparseThresholdLinear(
            in_dim=in_dim,
            out_dim=out_channels,
            fan_in=2 * fan_in_per_channel,
            weight_init=weight_init,
            theta_init=theta_init,
            idx=idx_global.reshape(out_channels, -1),
        )

    @staticmethod
    def _make_channel_pairs(in_channels: int, out_channels: int) -> torch.Tensor:
        """
        Each output selects exactly 2 input channels.
        """
        pairs = []
        for _ in range(out_channels):
            c = torch.randperm(in_channels)[:2]
            pairs.append(c)
        return torch.stack(pairs, dim=0).long()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, Cin=2, H, W)
        """
        B, _, H, W = x.shape

        x = F.pad(x, (self.padding,) * 4)

        Hout = (H + 2*self.padding - self.kernel_size) // self.stride + 1
        Wout = (W + 2*self.padding - self.kernel_size) // self.stride + 1

        patches = F.unfold(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride
        )  # (B, k*k*Cin, L)

        z = self.gatebank(patches)  # (B, Cout, L)

        z = einops.rearrange(z, "b c (h w) -> b c h w", h=Hout, w=Wout)

        return z

class SparseHierarchicalChannelLockedConv(nn.Module):
    """
    Hierarchical sparse logic convolution.

    For each output channel:
        - select EXACTLY 2 input channels
        - each selected channel has independent sparse taps
        - compute two partial convolutions
        - sum them

    Output:
        pre-activation z
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        *,
        fan_in_per_branch=6,
        weight_init="kaiming_uniform",
        theta_init="mean_abs_w",
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding

        k2 = kernel_size * kernel_size
        in_dim = in_channels * k2

        fan_in_per_branch = min(fan_in_per_branch, k2)

        # ---------------------------------------------------
        # 1) choose exactly 2 channels per output
        # ---------------------------------------------------
        chan_pairs = self._make_channel_pairs(
            in_channels,
            out_channels
        )

        self.register_buffer("chan_pairs", chan_pairs)

        # ===================================================
        # BRANCH 1
        # ===================================================

        idx1_local = SparseThresholdLinear._make_connections(
            k2,
            out_channels,
            fan_in_per_branch
        ).long()

        idx1_global = (
            chan_pairs[:, 0:1] * k2
            + idx1_local
        )

        self.branch1 = SparseThresholdLinear(
            in_dim=in_dim,
            out_dim=out_channels,
            fan_in=fan_in_per_branch,
            weight_init=weight_init,
            theta_init=theta_init,
            idx=idx1_global,
        )

        # ===================================================
        # BRANCH 2
        # ===================================================

        idx2_local = SparseThresholdLinear._make_connections(
            k2,
            out_channels,
            fan_in_per_branch
        ).long()

        idx2_global = (
            chan_pairs[:, 1:2] * k2
            + idx2_local
        )

        self.branch2 = SparseThresholdLinear(
            in_dim=in_dim,
            out_dim=out_channels,
            fan_in=fan_in_per_branch,
            weight_init=weight_init,
            theta_init=theta_init,
            idx=idx2_global,
        )

    @staticmethod
    def _make_channel_pairs(in_channels, out_channels):

        pairs = []

        for _ in range(out_channels):

            c = torch.randperm(in_channels)[:2]

            pairs.append(c)

        return torch.stack(pairs, dim=0).long()

    def forward(self, x):

        B, _, H, W = x.shape

        x = F.pad(x, (self.padding,) * 4)

        Hout = (
            H + 2*self.padding - self.kernel_size
        ) // self.stride + 1

        Wout = (
            W + 2*self.padding - self.kernel_size
        ) // self.stride + 1

        # ---------------------------------------------------
        # unfold
        # ---------------------------------------------------

        patches = F.unfold(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride
        )

        # ---------------------------------------------------
        # branch outputs
        # ---------------------------------------------------

        z1 = self.branch1(patches)

        z2 = self.branch2(patches)

        # hierarchical accumulation
        z = z1 + z2

        # reshape
        z = einops.rearrange(
            z,
            "b c (h w) -> b c h w",
            h=Hout,
            w=Wout
        )

        return z

class LogicMixedHierarchicalChannelLockedConv(nn.Module):
    """
    Logic-mixed hierarchical sparse logic convolution.

    For each output channel:
        - select EXACTLY 2 input channels
        - each selected channel has independent sparse taps (level 1)
        - apply STE to both level 1 outputs (binary features)
        - mix them using a 3rd learnable 2-input threshold gate (level 2)

    Output:
        pre-activation z (the result of the level-2 mixing gate)
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        *,
        fan_in_per_branch=6,
        weight_init="kaiming_uniform",
        theta_init="mean_abs_w",
        ste_scale=1.0,
        ste_window=1.0,
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding

        k2 = kernel_size * kernel_size
        in_dim = in_channels * k2
        fan_in_per_branch = min(fan_in_per_branch, k2)

        # ---------------------------------------------------
        # 1) choose exactly 2 channels per output
        # ---------------------------------------------------
        chan_pairs = self._make_channel_pairs(in_channels, out_channels)
        self.register_buffer("chan_pairs", chan_pairs)

        # ===================================================
        # LEVEL 1 BRANCHES
        # ===================================================
        idx1_local = SparseThresholdLinear._make_connections(k2, out_channels, fan_in_per_branch).long()
        idx1_global = chan_pairs[:, 0:1] * k2 + idx1_local
        self.branch1 = SparseThresholdLinear(
            in_dim=in_dim, out_dim=out_channels, fan_in=fan_in_per_branch,
            weight_init=weight_init, theta_init=theta_init, idx=idx1_global,
        )

        idx2_local = SparseThresholdLinear._make_connections(k2, out_channels, fan_in_per_branch).long()
        idx2_global = chan_pairs[:, 1:2] * k2 + idx2_local
        self.branch2 = SparseThresholdLinear(
            in_dim=in_dim, out_dim=out_channels, fan_in=fan_in_per_branch,
            weight_init=weight_init, theta_init=theta_init, idx=idx2_global,
        )

        # ===================================================
        # STE & LEVEL 2 MIXING GATE
        # ===================================================
        self.ste = StepGateClippedSTE(scale=ste_scale, w=ste_window)
        
        # 2nd level: each output j takes two binary inputs from its corresponding level-1 branches
        self.w_mix = nn.Parameter(torch.empty(out_channels, 2))
        self.theta_mix = nn.Parameter(torch.zeros(out_channels))
        
        self._init_weights(self.w_mix, scheme=weight_init)
        with torch.no_grad():
            self.theta_mix.copy_(self.w_mix.abs().mean(dim=1)) # Simple mean init

    @staticmethod
    def _make_channel_pairs(in_channels, out_channels):
        pairs = []
        for _ in range(out_channels):
            c = torch.randperm(in_channels)[:2]
            pairs.append(c)
        return torch.stack(pairs, dim=0).long()

    def _init_weights(self, w, scheme):
        if scheme == "kaiming_uniform":
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        elif scheme == "xavier_uniform":
            nn.init.xavier_uniform_(w)
        else:
            nn.init.normal_(w, mean=0.0, std=0.1)

    def forward(self, x):
        B, _, H, W = x.shape
        x = F.pad(x, (self.padding,) * 4)
        Hout = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        Wout = (W + 2 * self.padding - self.kernel_size) // self.stride + 1

        patches = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride)

        # Level 1: Sparse Sums
        z1 = self.branch1(patches)
        z2 = self.branch2(patches)

        # Level 1 Activations (Binarization)
        h1 = self.ste(z1)
        h2 = self.ste(z2)

        # Level 2: Mixing Logic
        # h1/h2: (B, Cout, L)
        h_combined = torch.stack([h1, h2], dim=2) # (B, Cout, 2, L)
        
        w = self.w_mix.view(1, self.out_channels, 2, 1)
        z_mix = (h_combined * w).sum(dim=2) - self.theta_mix.view(1, self.out_channels, 1) # (B, Cout, L)

        # Reshape to 2D spatial
        z = einops.rearrange(z_mix, "b c (h w) -> b c h w", h=Hout, w=Wout)
        return z
