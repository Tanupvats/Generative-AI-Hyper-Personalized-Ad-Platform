import math
import os
import torch
import torch.utils.data
import numpy as np
from librosa.filters import mel as librosa_mel_fn

MAX_WAV_VALUE = 32768.0


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    """
    Applies dynamic range compression to an input tensor.
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    """
    Reverses dynamic range compression.
    """
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    """
    Normalizes the magnitudes of a spectrogram.
    """
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    """
    De-normalizes the magnitudes of a spectrogram.
    """
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def mel_spectrogram_torch(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    """
    Calculates the mel spectrogram from an audio tensor using PyTorch STFT.
    """
    if torch.min(y) < -1.0 or torch.max(y) > 1.0:
        print(f"Warning: min value {torch.min(y)} max value {torch.max(y)}")

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)

    spec = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-9)

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec


def spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    """
    Calculates the linear spectrogram from an audio tensor.
    """
    if torch.min(y) < -1.0 or torch.max(y) > 1.0:
        print(f"Warning: min value {torch.min(y)} max value {torch.max(y)}")

    global hann_window
    if str(y.device) not in hann_window:
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)

    spec = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-9)

    return spec


def spec_to_mel_torch(spec, n_fft, num_mels, sampling_rate, fmin, fmax):
    """
    Converts an existing linear spectrogram tensor into a mel spectrogram tensor.
    """
    global mel_basis
    if str(fmax)+'_'+str(spec.device) not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(fmax)+'_'+str(spec.device)] = torch.from_numpy(mel).float().to(spec.device)

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(spec.device)], spec)
    spec = spectral_normalize_torch(spec)
    
    return spec