import os
import torch
import numpy as np
import librosa
from scipy.io import wavfile
from scipy import signal

from core.audio import load_audio
from core.slicer2 import Slicer
from core.pitch_extraction import get_f0_predictor

class DatasetPreprocessor:
    def __init__(self, sr=40000, device=None, is_half=True):
        self.sr = sr
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.is_half = is_half and "cuda" in self.device
        self.slicer = Slicer(sr=self.sr, threshold=-42, min_length=1500, min_interval=400, hop_size=15, max_sil_kept=500)
        self.bh, self.ah = signal.butter(N=5, Wn=48, btype="high", fs=self.sr)
        self.per, self.overlap, self.max_amp, self.alpha = 3.7, 0.3, 0.9, 0.75
        self.tail = self.per + self.overlap

    def process_dataset(self, input_dir, output_dir, hubert_model, f0_method="rmvpe"):
        wavs_gt_dir = os.path.join(output_dir, "0_gt_wavs")
        wavs_16k_dir = os.path.join(output_dir, "1_16k_wavs")
        f0_dir = os.path.join(output_dir, "2a_f0")
        f0nsf_dir = os.path.join(output_dir, "2b-f0nsf")
        feature_dir = os.path.join(output_dir, "3_feature256") 
        
        for d in [wavs_gt_dir, wavs_16k_dir, f0_dir, f0nsf_dir, feature_dir]: os.makedirs(d, exist_ok=True)
            
        f0_predictor = get_f0_predictor(f0_method, hop_length=160, sampling_rate=16000, device=self.device, is_half=self.is_half)

        for idx, filename in enumerate(os.listdir(input_dir)):
            if not filename.endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a')): continue
            try:
                audio = load_audio(os.path.join(input_dir, filename), self.sr)
                audio = signal.lfilter(self.bh, self.ah, audio)
                slice_idx = 0
                for audio_slice in self.slicer.slice(audio):
                    i = 0
                    while True:
                        start = int(self.sr * (self.per - self.overlap) * i)
                        i += 1
                        if len(audio_slice[start:]) > self.tail * self.sr:
                            self._norm_and_write(audio_slice[start : start + int(self.per * self.sr)], idx, slice_idx, wavs_gt_dir, wavs_16k_dir)
                            slice_idx += 1
                        else:
                            self._norm_and_write(audio_slice[start:], idx, slice_idx, wavs_gt_dir, wavs_16k_dir)
                            slice_idx += 1
                            break
            except Exception as e:
                print(f"Failed to slice {filename}: {e}")

        sliced_files = [f for f in os.listdir(wavs_16k_dir) if f.endswith('.wav')]
        
        with open(os.path.join(output_dir, "filelist.txt"), "w", encoding="utf-8") as f_list:
            for filename in sliced_files:
                wav_path = os.path.join(wavs_16k_dir, filename)
                base_name = filename.replace(".wav", "")
                try:
                    wav_16k, _ = librosa.load(wav_path, sr=16000)
                    p_len = wav_16k.shape[0] // 160
                    
                    featur_pit = f0_predictor.compute_f0(wav_16k, p_len)
                    np.save(os.path.join(f0nsf_dir, f"{base_name}.npy"), featur_pit, allow_pickle=False)
                    coarse_pit = self._coarse_f0(featur_pit)
                    np.save(os.path.join(f0_dir, f"{base_name}.npy"), coarse_pit, allow_pickle=False)

                    feats = torch.from_numpy(load_audio(wav_path, 16000)).float()
                    if feats.dim() == 2: feats = feats.mean(-1)
                    feats = feats.view(1, -1)
                    with torch.no_grad():
                        logits = hubert_model.extract_features(source=feats.half().to(self.device) if self.is_half else feats.to(self.device), padding_mask=torch.BoolTensor(feats.shape).fill_(False).to(self.device), output_layer=9)
                        final_feats = hubert_model.final_proj(logits[0]).squeeze(0).float().cpu().numpy()
                    np.save(os.path.join(feature_dir, f"{base_name}.npy"), final_feats, allow_pickle=False)

                    # Append to training filelist
                    f_list.write(f"{os.path.join(wavs_gt_dir, filename)}|{os.path.join(feature_dir, base_name+'.npy')}|{os.path.join(f0_dir, base_name+'.npy')}|{os.path.join(f0nsf_dir, base_name+'.npy')}|0\n")
                except Exception as e:
                    pass

    def _norm_and_write(self, tmp_audio, idx0, idx1, gt_dir, wavs16k_dir):
        tmp_max = np.abs(tmp_audio).max()
        if tmp_max > 2.5: return 
        tmp_audio = (tmp_audio / tmp_max * (self.max_amp * self.alpha)) + (1 - self.alpha) * tmp_audio
        wavfile.write(os.path.join(gt_dir, f"{idx0}_{idx1}.wav"), self.sr, tmp_audio.astype(np.float32))
        wavfile.write(os.path.join(wavs16k_dir, f"{idx0}_{idx1}.wav"), 16000, librosa.resample(tmp_audio, orig_sr=self.sr, target_sr=16000).astype(np.float32))

    def _coarse_f0(self, f0):
        f0_mel_min = 1127 * np.log(1 + 50.0 / 700)
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (1127 * np.log(1 + 1100.0 / 700) - f0_mel_min) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        return np.rint(f0_mel).astype(int)