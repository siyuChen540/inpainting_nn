from torch import nn

class DSConv2d(nn.Module):
    """
    Depthwise Separable Convolution.
    First performs a depthwise convolution, then a pointwise convolution.
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding=1, bias=False):
        super(DSConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class SELayer(nn.Module):
    """
    Squeeze-and-Excitation (SE) Layer.
    
    This layer adaptively recalibrates channel-wise feature responses by explicitly modeling interdependencies between channels.
    
    Args:
        num_channels (int): Number of input channels.
        reduction_ratio (int, optional): Reduction ratio for the bottleneck. Default is 16.
    """
    def __init__(self, num_channels, reduction_ratio=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(num_channels, num_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(num_channels // reduction_ratio, num_channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        batch_size, num_channels, _, _ = x.size()
        squeeze = self.avg_pool(x).view(batch_size, num_channels)
        excitation = self.fc(squeeze).view(batch_size, num_channels, 1, 1)
        scaled = x * excitation.expand_as(x)
        return scaled 