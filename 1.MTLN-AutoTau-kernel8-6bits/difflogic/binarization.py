import torch

class Thermometer:
    """
    Base Thermometer class for thermometer encoding.
    Methods:
    - fit(x): Fit the thermometer thresholds based on the data x.
    - binarize(x): Binarize the input x using the fitted thresholds.
    """
    def __init__(self, num_bits=1, feature_wise=True):
        
        assert num_bits > 0
        assert type(feature_wise) is bool

        self.num_bits = int(num_bits)
        self.feature_wise = feature_wise
        self.thresholds = None

    def get_thresholds(self, x):
        """
        Get thresholds for thermometer encoding.
        Args:   x: Input data to compute thresholds from.
        Returns: Computed thresholds.
        """
        min_value = x.min(dim=0)[0] if self.feature_wise else x.min()
        max_value = x.max(dim=0)[0] if self.feature_wise else x.max()
        return min_value.unsqueeze(-1) + torch.arange(1, self.num_bits+1).unsqueeze(0) * ((max_value - min_value) / (self.num_bits + 1)).unsqueeze(-1)

    def fit(self, x):
        """
        Fit the thermometer thresholds based on the data x.
        Args:   x: Input data to fit thresholds.
        Returns: self
        """
        if type(x) is not torch.Tensor:
            x = torch.tensor(x)
        self.thresholds = self.get_thresholds(x)
        return self
    
    def binarize(self, x, verbose=True):
        if self.thresholds is None:
            raise RuntimeError("need to fit before calling apply")
        if type(x) is not torch.Tensor:
            x = torch.tensor(x)
        
        N = x.shape[0]
        # Get shape from thresholds to pre-allocate
        if self.thresholds.ndim > 1:
            C, H, W, bits = self.thresholds.shape
            out_shape = (N, C, H, W, bits)
        else:
            bits = self.thresholds.shape[0]
            out_shape = (N, bits)
            
        if verbose:
            print(f"Binarizing {N} samples into {bits} bits...")
        out = torch.empty(out_shape, dtype=torch.bool)
        
        chunk_size = 500
        for i in range(0, N, chunk_size):
            end = min(i + chunk_size, N)
            x_chunk = x[i:end].unsqueeze(-1)
            out[i:end] = (x_chunk > self.thresholds)
            if verbose and i % 1000 == 0:
                print(f"  Binarized {i}/{N}...")
        
        return out

class GaussianThermometer(Thermometer):
    """
    A specialized thermometer that uses Gaussian quantiles as thresholds.
    This thermometer computes thresholds based on the mean and standard deviation
    of the input data, scaled by the inverse CDF of a standard normal distribution.
    """
    def __init__(self, num_bits=1, feature_wise=True):
        super().__init__(num_bits, feature_wise)

    def get_thresholds(self, x):
        std_skews = torch.distributions.Normal(0, 1).icdf(torch.arange(1, self.num_bits+1)/(self.num_bits+1))
        mean = x.mean(dim=0) if self.feature_wise else x.mean()
        std = x.std(dim=0) if self.feature_wise else x.std() 
        thresholds = torch.stack([std_skew * std + mean for std_skew in std_skews], dim=-1)
        return thresholds
    
    
class DistributiveThermometer(Thermometer):
    """
    A specialized thermometer that uses data distribution quantiles as thresholds.
    This thermometer computes thresholds based on the sorted values of the input data,
    selecting quantiles that divide the data into equal parts.
    """
    def __init__(self, num_bits=1, feature_wise=True):
        super().__init__(num_bits, feature_wise)

    def get_thresholds(self, x):
        print("Calculating thresholds using quantiles...")
        # Use a subset of data if it's too large to speed up quantile calculation
        max_samples = 2000
        if x.shape[0] > max_samples:
            indices = torch.randperm(x.shape[0])[:max_samples]
            x_input = x[indices]
        else:
            x_input = x

        # Use quantiles instead of full sort to save memory
        q = torch.linspace(0, 1, self.num_bits + 2)[1:-1].to(x.device).to(x.dtype)
        if not self.feature_wise:
            thresholds = torch.quantile(x_input.flatten(), q)
            return thresholds
        else:
            # torch.quantile across batch dimension
            thresholds = torch.quantile(x_input, q, dim=0)
            # thresholds shape is [num_bits, channels, H, W]
            # Need to match the expected return shape: [channels, H, W, num_bits]
            return torch.permute(thresholds, (*list(range(1, thresholds.ndim)), 0))

