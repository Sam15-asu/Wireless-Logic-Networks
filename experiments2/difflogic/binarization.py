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
    
    def binarize(self, x):
        if self.thresholds is None:
            raise RuntimeError("need to fit before calling apply")
        if type(x) is not torch.Tensor:
            x = torch.tensor(x)
        x = x.unsqueeze(-1)
        return (x > self.thresholds)   # keep as bool

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
        data = torch.sort(x.flatten())[0] if not self.feature_wise else torch.sort(x, dim=0)[0]
        indicies = torch.tensor([int(data.shape[0]*i/(self.num_bits+1)) for i in range(1, self.num_bits+1)])
        thresholds = data[indicies]
        return torch.permute(thresholds, (*list(range(1, thresholds.ndim)), 0))

