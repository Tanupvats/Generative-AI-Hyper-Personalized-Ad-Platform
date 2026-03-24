import math
from typing import Optional

import torch
from torch.nn import functional as F


def init_weights(m, mean=0.0, std=0.01):
    """Initializes weights for Convolutional layers."""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    """Calculates padding to maintain sequence length during 1D Convolutions."""
    return int((kernel_size * dilation - dilation) / 2)


def convert_pad_shape(pad_shape):
    """Flattens a 2D padding shape array for PyTorch's F.pad"""
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape


def kl_divergence(m_p, logs_p, m_q, logs_q):
    """Calculates the Kullback-Leibler divergence between two Gaussians."""
    kl = (logs_q - logs_p) - 0.5
    kl += 0.5 * (torch.exp(2.0 * logs_p) + ((m_p - m_q) ** 2)) * torch.exp(-2.0 * logs_q)
    return kl


def rand_gumbel(shape):
    """Sample from the Gumbel distribution, protect from overflows."""
    uniform_samples = torch.rand(shape) * 0.99998 + 0.00001
    return -torch.log(-torch.log(uniform_samples))


def rand_gumbel_like(x):
    g = rand_gumbel(x.size()).to(dtype=x.dtype, device=x.device)
    return g


def slice_segments(x, ids_str, segment_size=4):
    """
    Slices segments from a 3D tensor [Batch, Channels, Time] using starting indices.
    """
    ret = torch.zeros_like(x[:, :, :segment_size])
    for i in range(x.size(0)):
        idx_str = ids_str[i]
        idx_end = idx_str + segment_size
        ret[i] = x[i, :, idx_str:idx_end]
    return ret


def slice_segments2(x, ids_str, segment_size=4):
    """
    Slices segments from a 2D tensor [Batch, Time] using starting indices.
    """
    ret = torch.zeros_like(x[:, :segment_size])
    for i in range(x.size(0)):
        idx_str = ids_str[i]
        idx_end = idx_str + segment_size
        ret[i] = x[i, idx_str:idx_end]
    return ret


def rand_slice_segments(x, x_lengths=None, segment_size=4):
    """
    Randomly selects a starting index and slices a segment of `segment_size` from a tensor.
    Used extensively during the VITS training loop to compare chunks of generated vs real audio.
    """
    b, d, t = x.size()
    if x_lengths is None:
        x_lengths = t
    ids_str_max = x_lengths - segment_size + 1
    ids_str_max = torch.clamp(ids_str_max, min=1) # Prevent <= 0 bounds
    ids_str = (torch.rand([b]).to(device=x.device) * ids_str_max).to(dtype=torch.long)
    ret = slice_segments(x, ids_str, segment_size)
    return ret, ids_str


def sequence_mask(length: torch.Tensor, max_length: Optional[int] = None):
    """
    Creates a boolean mask from sequence lengths.
    Example: lengths=[2, 3], max_length=4 -> [[T, T, F, F], [T, T, T, F]]
    """
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def generate_path(duration, mask):
    """
    Generates a monotonic path for alignment.
    duration: [b, 1, t_x]
    mask: [b, 1, t_y, t_x]
    """
    device = duration.device

    b, _, t_y, t_x = mask.shape
    cum_duration = torch.cumsum(duration, -1)

    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
    path = path.view(b, t_x, t_y)
    path = path - F.pad(path, convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:, :-1]
    path = path.unsqueeze(1).transpose(2, 3) * mask
    return path


def clip_grad_value_(parameters, clip_value, norm_type=2):
    """
    Clips the gradients of the model parameters to prevent exploding gradients during training.
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if clip_value is not None:
        clip_value = float(clip_value)

    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
        if clip_value is not None:
            p.grad.data.clamp_(min=-clip_value, max=clip_value)
    total_norm = total_norm ** (1.0 / norm_type)
    return total_norm


def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    """
    Fused operation used heavily in the WaveNet (WN) residual blocks.
    Splits the channels in half, applies Tanh to the first half, Sigmoid to the second,
    and multiplies them together.
    """
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts