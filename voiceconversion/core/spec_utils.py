import numpy as np
import librosa
import torch


def wave_to_spectrogram(wave, hop_length, n_fft):
    """
    Converts a multi-channel waveform into a complex spectrogram.
    """
    if wave.ndim == 1:
        wave = np.array([wave, wave])
        
    spec_left = librosa.stft(wave[0], n_fft=n_fft, hop_length=hop_length)
    spec_right = librosa.stft(wave[1], n_fft=n_fft, hop_length=hop_length)
    
    spec = np.ascontiguousarray([spec_left, spec_right])
    return spec


def combine_spectrograms(specs, mp):
    """
    Combines multi-band spectrograms into a single representation based on 
    the model parameters (mp).
    """
    n_bins = mp.param["n_fft"] // 2 + 1
    n_frames = specs[len(specs)].shape[2]
    combined_spec = np.zeros((2, n_bins, n_frames), dtype=np.complex64)
    
    for d in range(len(specs), 0, -1):
        bp = mp.param["band"][d]
        spec = specs[d]
        
        if d == len(specs):
            combined_spec[:, bp["res_l"]:bp["res_h"], :] = spec[:, bp["res_l"]:bp["res_h"], :]
        else:
            # For lower bands, we might need to pad/crop frames to match the high-band timing
            spec_frames = spec.shape[2]
            if spec_frames > n_frames:
                spec = spec[:, :, :n_frames]
            elif spec_frames < n_frames:
                spec = np.pad(spec, ((0, 0), (0, 0), (0, n_frames - spec_frames)), mode='constant')
            
            combined_spec[:, bp["res_l"]:bp["res_h"], :] = spec[:, bp["res_l"]:bp["res_h"], :]
            
    return combined_spec


def cmb_spectrogram_to_wave(spec, mp):
    """
    Converts a combined complex spectrogram back into a time-domain waveform 
    using the Inverse Short-Time Fourier Transform (ISTFT).
    """
    vocal_wave = []
    for d in range(len(mp.param["band"]), 0, -1):
        bp = mp.param["band"][d]
        
        # Extract band-specific bins
        spec_band = np.zeros((2, bp["n_fft"] // 2 + 1, spec.shape[2]), dtype=np.complex64)
        spec_band[:, bp["res_l"]:bp["res_h"], :] = spec[:, bp["res_l"]:bp["res_h"], :]
        
        # ISTFT for each channel
        wav_l = librosa.istft(spec_band[0], hop_length=bp["hl"], win_length=bp["n_fft"])
        wav_r = librosa.istft(spec_band[1], hop_length=bp["hl"], win_length=bp["n_fft"])
        
        wave = np.array([wav_l, wav_r])
        
        if d == len(mp.param["band"]):
            vocal_wave = wave
        else:
            # Resample and add to the cumulative wave
            wave_resampled = librosa.resample(
                wave, 
                orig_sr=bp["sr"], 
                target_sr=mp.param["sr"]
            )
            # Ensure lengths match before adding
            if wave_resampled.shape[1] > vocal_wave.shape[1]:
                wave_resampled = wave_resampled[:, :vocal_wave.shape[1]]
            elif wave_resampled.shape[1] < vocal_wave.shape[1]:
                wave_resampled = np.pad(wave_resampled, ((0, 0), (0, vocal_wave.shape[1] - wave_resampled.shape[1])), mode='constant')
            
            vocal_wave += wave_resampled
            
    return vocal_wave.T # Return as [Time, Channels] for soundfile saving


def spec_to_mag_phase(spec):
    """
    Decomposes a complex spectrogram into magnitude and phase components.
    """
    mag = np.abs(spec)
    phase = np.exp(1.j * np.angle(spec))
    return mag, phase


def mag_phase_to_spec(mag, phase):
    """
    Recomposes a complex spectrogram from magnitude and phase components.
    """
    return mag * phase