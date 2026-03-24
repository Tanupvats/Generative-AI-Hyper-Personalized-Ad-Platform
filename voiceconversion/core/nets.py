import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2DBnRelu(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        bias,
        padding_mode="zeros",
    ):
        super(Conv2DBnRelu, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            padding_mode=padding_mode,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ASPPModule(nn.Module):
    def __init__(self, in_channels, out_channels, dilations):
        super(ASPPModule, self).__init__()
        self.conv1 = Conv2DBnRelu(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            bias=False,
        )
        self.conv2 = Conv2DBnRelu(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=dilations[0],
            dilation=dilations[0],
            bias=False,
            padding_mode="replicate",
        )
        self.conv3 = Conv2DBnRelu(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=dilations[1],
            dilation=dilations[1],
            bias=False,
            padding_mode="replicate",
        )
        self.conv4 = Conv2DBnRelu(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=dilations[2],
            dilation=dilations[2],
            bias=False,
            padding_mode="replicate",
        )

        self.bottleneck = Conv2DBnRelu(
            out_channels * 4,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            bias=False,
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        
        out = torch.cat((x1, x2, x3, x4), dim=1)
        out = self.bottleneck(out)
        return out


class CascadedASPPNet(nn.Module):
    """
    The main UVR5 model architecture for vocal separation.
    Uses Cascaded Atrous Spatial Pyramid Pooling to capture multi-scale context.
    """
    def __init__(self, n_fft):
        super(CascadedASPPNet, self).__init__()
        self.n_fft = n_fft
        
        # Primary feature extraction block
        self.conv1 = Conv2DBnRelu(
            in_channels=2, # Stereo spectrogram (Real + Imaginary/Phase info or L/R channels)
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            bias=False,
            padding_mode="replicate",
        )
        
        # Cascaded ASPP Modules with increasing dilation rates
        self.aspp1 = ASPPModule(
            in_channels=32, out_channels=64, dilations=[2, 4, 6]
        )
        self.aspp2 = ASPPModule(
            in_channels=64, out_channels=128, dilations=[4, 8, 12]
        )
        self.aspp3 = ASPPModule(
            in_channels=128, out_channels=256, dilations=[8, 16, 24]
        )
        self.aspp4 = ASPPModule(
            in_channels=256, out_channels=256, dilations=[16, 32, 48]
        )
        
        # Decoding / Mask Generation Block
        self.conv2 = Conv2DBnRelu(
            in_channels=256,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            bias=False,
            padding_mode="replicate",
        )
        self.conv3 = Conv2DBnRelu(
            in_channels=128,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            bias=False,
            padding_mode="replicate",
        )
        self.conv4 = Conv2DBnRelu(
            in_channels=64,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            bias=False,
            padding_mode="replicate",
        )
        self.conv5 = nn.Conv2d(
            in_channels=32,
            out_channels=2,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            bias=False,
            padding_mode="replicate",
        )
        
        self.out = nn.Sigmoid()

    def forward(self, x):
        """
        Input: Mixed spectrogram tensor
        Output: Predicted mask to isolate vocals/instruments
        """
        x = self.conv1(x)
        x = self.aspp1(x)
        x = self.aspp2(x)
        x = self.aspp3(x)
        x = self.aspp4(x)
        
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        return self.out(x)