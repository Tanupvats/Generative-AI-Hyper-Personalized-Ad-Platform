import os
import torch
import torch.nn.functional as F
import numpy as np
import faiss
from fairseq import checkpoint_utils

# Import our decoupled ecosystem modules
from core.audio import load_audio
from core.slicer2 import Slicer
from core.pitch_extraction import get_f0_predictor
from core.models import SynthesizerTrnMs256NSFsid, SynthesizerTrnMs768NSFsid

class RVCEngine:
    def __init__(self, device=None, is_half=None):
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Use half-precision (FP16) on CUDA to save memory and boost speed
        if is_half is None:
            self.is_half = "cuda" in self.device
        else:
            self.is_half = is_half

        self.hubert_model = None
        self.vits_model = None
        self.faiss_index = None
        self.big_npy = None
        self.tgt_sr = 40000  # Default, overwritten when a model is loaded

    def load_hubert(self, hubert_path):
        """Loads the Fairseq HuBERT model for semantic feature extraction."""
        models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
            [hubert_path], 
            suffix=""
        )
        self.hubert_model = models[0].to(self.device)
        if self.is_half:
            self.hubert_model = self.hubert_model.half()
        self.hubert_model.eval()

    def load_vits(self, model_path):
        """Loads the VITS Generator dynamically, auto-detecting V1 vs V2."""
        cpt = torch.load(model_path, map_location="cpu")
        self.tgt_sr = cpt["config"][-1]
        
        # Determine the number of speakers from the embedding weight
        cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0] 
        
        # Check architecture version (768 means V2, otherwise V1)
        if cpt["config"][2] == 768:
            self.vits_model = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=self.is_half)
        else:
            self.vits_model = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=self.is_half)
            
        self.vits_model.load_state_dict(cpt["weight"], strict=False)
        self.vits_model.eval().to(self.device)
        
        if self.is_half:
            self.vits_model = self.vits_model.half()

    def load_faiss_index(self, index_path):
        """Loads the Faiss Index for timbre/accent matching."""
        self.faiss_index = faiss.read_index(index_path)
        self.big_npy = self.faiss_index.reconstruct_n(0, self.faiss_index.ntotal)

    def infer(self, audio_path, f0_up_key, f0_method="rmvpe", index_rate=0.75):
        """
        The core chunked inference loop.
        Uses Slicer to prevent CUDA Out-of-Memory (OOM) errors on long audio files.
        """
        # 1. Load and resample audio to 16kHz for HuBERT
        audio = load_audio(audio_path, sr=16000)
        
        # 2. Slice audio on silences
        slicer = Slicer(
            sr=16000, 
            threshold=-40, 
            min_length=5000, 
            min_interval=300, 
            hop_size=10, 
            max_sil_kept=500
        )
        chunks = slicer.slice(audio)
        audio_outputs = []
        
        # 3. Instantiate the requested pitch predictor (RMVPE, Harvest, PM, or DIO)
        f0_predictor = get_f0_predictor(
            f0_method, 
            hop_length=160, 
            sampling_rate=16000, 
            device=self.device, 
            is_half=self.is_half
        )
        
        # 4. Process each chunk sequentially
        for chunk in chunks:
            # Skip micro-chunks (less than 0.1 seconds) to prevent conv layer crashes
            if len(chunk) < int(16000 * 0.1): 
                audio_outputs.append(np.zeros(int(self.tgt_sr * (len(chunk)/16000)), dtype=np.float32))
                continue
                
            chunk_out = self._infer_chunk(chunk, f0_up_key, f0_predictor, index_rate)
            audio_outputs.append(chunk_out)

        # 5. Concatenate back into a single continuous stream
        return np.concatenate(audio_outputs), self.tgt_sr

    def _infer_chunk(self, audio_chunk, f0_up_key, f0_predictor, index_rate):
        """Processes a single slice of audio through the HuBERT -> Faiss -> VITS pipeline."""
        feats = torch.from_numpy(audio_chunk).float().view(1, -1)
        padding_mask = torch.BoolTensor(feats.shape).fill_(False)
        
        # Determine target layer based on V1 vs V2 architecture
        is_v2 = hasattr(self.vits_model, 'enc_p') and self.vits_model.enc_p.in_channels == 768
        output_layer = 12 if is_v2 else 9
        
        inputs = {
            "source": feats.half().to(self.device) if self.is_half else feats.to(self.device),
            "padding_mask": padding_mask.to(self.device),
            "output_layer": output_layer 
        }
        
        # --- A. Extract Semantic Features (HuBERT) ---
        with torch.no_grad():
            logits = self.hubert_model.extract_features(**inputs)
            # V1 uses final_proj on layer 9, V2 uses raw layer 12
            feats = self.hubert_model.final_proj(logits[0]) if output_layer == 9 else logits[0]

        # --- B. Timbre Blending (Faiss Index) ---
        if self.faiss_index is not None and index_rate > 0:
            npy = feats[0].cpu().numpy().astype("float32")
            score, ix = self.faiss_index.search(npy, k=8)
            
            weight = np.square(1 / score)
            weight /= weight.sum(axis=1, keepdims=True)
            npy_reconstructed = np.sum(self.big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)
            
            if self.is_half:
                npy_reconstructed = npy_reconstructed.astype("float16")
                
            # Mix the retrieved timbre features with the original semantic features
            feats[0] = (torch.from_numpy(npy_reconstructed).unsqueeze(0).to(self.device) * index_rate 
                        + (1 - index_rate) * feats[0])

        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        p_len = min(feats.shape[1], 10000)
        
        # --- C. Extract and Quantize Pitch (F0) ---
        pitch = f0_predictor.compute_f0(audio_chunk, p_len)
        pitch *= pow(2, f0_up_key / 12)
        pitchf = pitch.copy()
        
        # Convert Hertz to Mel scale bins (1 to 255) for the VITS embedding table
        f0_mel_min = 1127 * np.log(1 + 50.0 / 700)
        f0_mel_max = 1127 * np.log(1 + 1100.0 / 700)
        
        f0_mel = 1127 * np.log(1 + pitch / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        pitch_coarse = np.rint(f0_mel).astype(np.int64)

        # --- D. Final VITS Generation ---
        p_len = min(feats.shape[1], 10000, pitch.shape[0])
        feats = feats[:, :p_len, :]
        
        p_len_tensor = torch.LongTensor([p_len]).to(self.device)
        pitch_tensor = torch.LongTensor(pitch_coarse[:p_len]).unsqueeze(0).to(self.device)
        pitchf_tensor = torch.FloatTensor(pitchf[:p_len]).unsqueeze(0).to(self.device)
        sid_tensor = torch.LongTensor([0]).to(self.device)
        
        with torch.no_grad():
            # Infer through the VITS Generator
            audio_out = self.vits_model.infer(
                feats, 
                p_len_tensor, 
                pitch_tensor, 
                pitchf_tensor, 
                sid_tensor
            )[0][0, 0]
            
        return audio_out.data.cpu().float().numpy()