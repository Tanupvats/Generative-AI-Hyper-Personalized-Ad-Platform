import numpy as np
import parselmouth
import pyworld
from core.rmvpe import RMVPE

class F0Predictor:
    def __init__(self, hop_length=512, f0_min=50, f0_max=1100, sampling_rate=40000):
        self.hop_length, self.f0_min, self.f0_max, self.sampling_rate = hop_length, f0_min, f0_max, sampling_rate

    def interpolate_f0(self, f0):
        data = np.reshape(f0, (f0.size, 1))
        vuv_vector = np.zeros((data.size, 1), dtype=np.float32)
        vuv_vector[data > 0.0] = 1.0
        ip_data = data.copy()
        last_value = 0.0
        for i in range(data.size):
            if data[i] <= 0.0:
                j = i + 1
                for j in range(i + 1, data.size):
                    if data[j] > 0.0: break
                if j < data.size - 1:
                    if last_value > 0.0:
                        step = (data[j] - data[i - 1]) / float(j - i)
                        for k in range(i, j): ip_data[k] = data[i - 1] + step * (k - i + 1)
                    else:
                        for k in range(i, j): ip_data[k] = data[j]
                else:
                    for k in range(i, data.size): ip_data[k] = last_value
            else:
                last_value = data[i]
        return ip_data[:, 0], vuv_vector[:, 0]

class PMF0Predictor(F0Predictor):
    def compute_f0(self, wav, p_len):
        f0 = parselmouth.Sound(wav, self.sampling_rate).to_pitch_ac(
            time_step=(self.hop_length / self.sampling_rate), voicing_threshold=0.6, pitch_floor=self.f0_min, pitch_ceiling=self.f0_max
        ).selected_array["frequency"]
        pad_size = (p_len - len(f0) + 1) // 2
        if pad_size > 0 or p_len - len(f0) - pad_size > 0: f0 = np.pad(f0, [[pad_size, p_len - len(f0) - pad_size]], mode="constant")
        return self.interpolate_f0(f0)[0]

class HarvestF0Predictor(F0Predictor):
    def compute_f0(self, wav, p_len):
        f0, t = pyworld.harvest(wav.astype(np.double), fs=self.sampling_rate, f0_ceil=self.f0_max, f0_floor=self.f0_min, frame_period=1000 * self.hop_length / self.sampling_rate)
        f0 = pyworld.stonemask(wav.astype(np.double), f0, t, self.sampling_rate)
        source = np.array(f0)
        source[source < 0.001] = np.nan
        target = np.interp(np.arange(0, len(source) * p_len, len(source)) / p_len, np.arange(0, len(source)), source)
        return self.interpolate_f0(np.nan_to_num(target))[0]

class DioF0Predictor(HarvestF0Predictor):
    def compute_f0(self, wav, p_len):
        f0, t = pyworld.dio(wav.astype(np.double), fs=self.sampling_rate, f0_floor=self.f0_min, f0_ceil=self.f0_max, frame_period=1000 * self.hop_length / self.sampling_rate)
        f0 = pyworld.stonemask(wav.astype(np.double), f0, t, self.sampling_rate)
        source = np.array(f0)
        source[source < 0.001] = np.nan
        target = np.interp(np.arange(0, len(source) * p_len, len(source)) / p_len, np.arange(0, len(source)), source)
        return self.interpolate_f0(np.nan_to_num(target))[0]

def get_f0_predictor(method, hop_length=512, sampling_rate=40000, device="cpu", is_half=True):
    if method == "pm": return PMF0Predictor(hop_length=hop_length, sampling_rate=sampling_rate)
    elif method == "harvest": return HarvestF0Predictor(hop_length=hop_length, sampling_rate=sampling_rate)
    elif method == "dio": return DioF0Predictor(hop_length=hop_length, sampling_rate=sampling_rate)
    elif method == "rmvpe": return RMVPE("assets/rmvpe/rmvpe.pt", is_half=is_half, device=device)
    raise ValueError(f"Unknown method: {method}")