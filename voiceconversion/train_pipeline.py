import os
import logging
from random import randint
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
import numpy as np
import faiss
from sklearn.cluster import MiniBatchKMeans

from core.data_utils import TextAudioLoaderMultiNSFsid, TextAudioCollateMultiNSFsid, DistributedBucketSampler
from core.models import SynthesizerTrnMs768NSFsid, MultiPeriodDiscriminatorV2
from core.losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from core.mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from core.commons import slice_segments, clip_grad_value_

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

@dataclass
class TrainConfig:
    dataset_filelist: str
    model_save_dir: str
    epochs: int = 100
    batch_size: int = 8
    learning_rate: float = 1e-4
    lr_decay: float = 0.999875
    segment_size: int = 17280
    c_mel: int = 45
    c_kl: float = 1.0
    fp16_run: bool = True
    seed: int = 1234
    log_interval: int = 200
    sampling_rate: int = 40000
    filter_length: int = 1024
    hop_length: int = 400
    win_length: int = 1024
    n_mel_channels: int = 128
    mel_fmin: float = 0.0
    mel_fmax: float = None

def train_faiss_index(feature_dir, out_path):
    print("Training Faiss Index...")
    npys = [np.load(os.path.join(feature_dir, f)) for f in os.listdir(feature_dir) if f.endswith('.npy')]
    big_npy = np.concatenate(npys, axis=0)
    if big_npy.shape[0] > 200000:
        big_npy = MiniBatchKMeans(n_clusters=10000, batch_size=8192).fit(big_npy).cluster_centers_
    n_ivf = max(min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39), 1)
    index = faiss.index_factory(big_npy.shape[1], f"IVF{n_ivf},Flat")
    index_ivf = faiss.extract_index_ivf(index)
    index_ivf.nprobe = 1
    index.train(big_npy)
    index.add(big_npy)
    faiss.write_index(index, out_path)
    print(f"Index saved to {out_path}")

def run_training_worker(rank, n_gpus, config: TrainConfig):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(randint(20000, 55555))
    dist.init_process_group(backend="gloo" if os.name == "nt" else "nccl", init_method="env://", world_size=n_gpus, rank=rank)
    
    torch.manual_seed(config.seed)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available(): torch.cuda.set_device(rank)

    train_dataset = TextAudioLoaderMultiNSFsid(config.dataset_filelist, config)
    train_sampler = DistributedBucketSampler(
        train_dataset, config.batch_size * n_gpus, [100, 200, 300, 400, 500, 600, 700, 800, 900], num_replicas=n_gpus, rank=rank, shuffle=True
    )
    collate_fn = TextAudioCollateMultiNSFsid()
    train_loader = DataLoader(train_dataset, num_workers=4, shuffle=False, pin_memory=True, collate_fn=collate_fn, batch_sampler=train_sampler)

    net_g = SynthesizerTrnMs768NSFsid(
        spec_channels=config.filter_length // 2 + 1, segment_size=config.segment_size // config.hop_length, inter_channels=192, hidden_channels=192, filter_channels=768, n_heads=2, n_layers=6, kernel_size=3, p_dropout=0, resblock="1", resblock_kernel_sizes=[3, 7, 11], resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]], upsample_rates=[10, 10, 2, 2], upsample_initial_channel=512, upsample_kernel_sizes=[16, 16, 4, 4], spk_embed_dim=109, gin_channels=256, sr=config.sampling_rate, is_half=config.fp16_run
    ).to(device)
    net_d = MultiPeriodDiscriminatorV2(use_spectral_norm=False).to(device)

    net_g = DDP(net_g, device_ids=[rank]) if torch.cuda.is_available() else DDP(net_g)
    net_d = DDP(net_d, device_ids=[rank]) if torch.cuda.is_available() else DDP(net_d)

    optim_g = torch.optim.AdamW(net_g.parameters(), lr=config.learning_rate, betas=[0.8, 0.99])
    optim_d = torch.optim.AdamW(net_d.parameters(), lr=config.learning_rate, betas=[0.8, 0.99])
    
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=config.lr_decay)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=config.lr_decay)
    scaler = GradScaler(enabled=config.fp16_run)

    global_step = 0
    for epoch in range(1, config.epochs + 1):
        train_loader.batch_sampler.set_epoch(epoch)
        net_g.train()
        net_d.train()
        
        for batch_idx, batch in enumerate(train_loader):
            phone, phone_lengths, pitch, pitchf, spec, spec_lengths, wave, wave_lengths, sid = [x.to(device, non_blocking=True) for x in batch]
            
            with autocast(enabled=config.fp16_run):
                y_hat, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid)
                wave_slice = slice_segments(wave, ids_slice * config.hop_length, config.segment_size)
                y_d_hat_r, y_d_hat_g, _, _ = net_d(wave_slice, y_hat.detach())
                with autocast(enabled=False): loss_disc, _, _ = discriminator_loss(y_d_hat_r, y_d_hat_g)
                    
            optim_d.zero_grad()
            scaler.scale(loss_disc).backward()
            scaler.unscale_(optim_d)
            clip_grad_value_(net_d.parameters(), None)
            scaler.step(optim_d)

            with autocast(enabled=config.fp16_run):
                y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(wave_slice, y_hat)
                mel = spec_to_mel_torch(spec, config.filter_length, config.n_mel_channels, config.sampling_rate, config.mel_fmin, config.mel_fmax)
                y_mel = slice_segments(mel, ids_slice, config.segment_size // config.hop_length)
                
                with autocast(enabled=False):
                    y_hat_mel = mel_spectrogram_torch(y_hat.float().squeeze(1), config.filter_length, config.n_mel_channels, config.sampling_rate, config.hop_length, config.win_length, config.mel_fmin, config.mel_fmax)
                    if config.fp16_run: y_hat_mel = y_hat_mel.half()
                    loss_mel = F.l1_loss(y_mel, y_hat_mel) * config.c_mel
                    loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * config.c_kl
                    loss_fm = feature_loss(fmap_r, fmap_g)
                    loss_gen, _ = generator_loss(y_d_hat_g)
                    loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl

            optim_g.zero_grad()
            scaler.scale(loss_gen_all).backward()
            scaler.unscale_(optim_g)
            clip_grad_value_(net_g.parameters(), None)
            scaler.step(optim_g)
            scaler.update()

            if rank == 0 and global_step % config.log_interval == 0:
                logger.info(f"Epoch: {epoch} | Step: {global_step} | Gen Loss: {loss_gen_all.item():.4f} | Disc Loss: {loss_disc.item():.4f}")
            global_step += 1

        scheduler_g.step()
        scheduler_d.step()
        
        if rank == 0 and epoch % 10 == 0:
            g_state = net_g.module.state_dict() if hasattr(net_g, "module") else net_g.state_dict()
            d_state = net_d.module.state_dict() if hasattr(net_d, "module") else net_d.state_dict()
            torch.save({"model": g_state, "config": config.__dict__}, os.path.join(config.model_save_dir, f"G_{epoch}.pth"))
            torch.save({"model": d_state, "config": config.__dict__}, os.path.join(config.model_save_dir, f"D_{epoch}.pth"))

def trigger_training(dataset_filelist: str, model_save_dir: str, **kwargs):
    os.makedirs(model_save_dir, exist_ok=True)
    config = TrainConfig(dataset_filelist=dataset_filelist, model_save_dir=model_save_dir, **kwargs)
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    mp.spawn(run_training_worker, nprocs=n_gpus, args=(n_gpus, config))