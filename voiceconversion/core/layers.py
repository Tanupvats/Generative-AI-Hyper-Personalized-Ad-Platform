import torch
from torch import nn

class LinearNorm(torch.nn.Module):
    """
    Linear layer with Xavier uniform initialization.
    Helps prevent vanishing/exploding gradients in deep networks.
    """
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain)
        )

    def forward(self, x):
        return self.linear_layer(x)


class ConvNorm(torch.nn.Module):
    """
    1D Convolutional layer with Xavier uniform initialization and 
    automatic padding calculation to maintain sequence length.
    """
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size=1, 
        stride=1,
        padding=None, 
        dilation=1, 
        bias=True, 
        w_init_gain='linear'
    ):
        super(ConvNorm, self).__init__()
        
        # Auto-calculate padding to keep sequence length consistent
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(
            in_channels, 
            out_channels,
            kernel_size=kernel_size, 
            stride=stride,
            padding=padding, 
            dilation=dilation,
            bias=bias
        )

        torch.nn.init.xavier_uniform_(
            self.conv.weight, 
            gain=torch.nn.init.calculate_gain(w_init_gain)
        )

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class Highway(nn.Module):
    """
    Highway Network Layer.
    Allows for training very deep networks by establishing direct pathways 
    for information flow, gated by a learned transform/carry mechanism.
    """
    def __init__(self, in_size, out_size):
        super(Highway, self).__init__()
        self.H = nn.Linear(in_size, out_size)
        self.H.bias.data.zero_()
        self.T = nn.Linear(in_size, out_size)
        self.T.bias.data.fill_(-1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        H = self.relu(self.H(inputs))
        T = self.sigmoid(self.T(inputs))
        return H * T + inputs * (1.0 - T)