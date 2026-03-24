import os
import torch
from torch import optim
from torch.utils import data as data_utils

# Import core models and utils from original codebase
from models import Wav2Lip, Wav2Lip_disc_qual
from models import SyncNet_color as SyncNet
from hparams import hparams
from hq_wav2lip_train import Dataset, train, load_checkpoint

def run_hq_training(data_root, checkpoint_dir, syncnet_checkpoint, device):
    """
    Wrapper for hq_wav2lip_train.py that can be invoked via the API background task.
    Initializes models, dataloaders, and runs the training loop.
    """
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Mock the args object that the original script expects
    class Args:
        pass
    args = Args()
    args.data_root = data_root

    train_dataset = Dataset('train', args)
    test_dataset = Dataset('val', args)

    train_data_loader = data_utils.DataLoader(
        train_dataset, batch_size=hparams.batch_size, shuffle=True,
        num_workers=hparams.num_workers
    )

    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=hparams.batch_size,
        num_workers=4
    )

    model = Wav2Lip().to(device)
    disc = Wav2Lip_disc_qual().to(device)
    syncnet = SyncNet().to(device)

    # Freeze SyncNet parameters (it's the expert discriminator)
    for p in syncnet.parameters():
        p.requires_grad = False

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=hparams.initial_learning_rate, betas=(0.5, 0.999))
    disc_optimizer = optim.Adam([p for p in disc.parameters() if p.requires_grad],
                           lr=hparams.disc_initial_learning_rate, betas=(0.5, 0.999))
        
    # Load the pretrained expert discriminator
    try:
        load_checkpoint(syncnet_checkpoint, syncnet, None, reset_optimizer=True, overwrite_global_states=False)
        print("Loaded SyncNet Expert Discriminator.")
    except Exception as e:
        print(f"Failed to load SyncNet: {e}. Ensure checkpoint exists at {syncnet_checkpoint}")
        return

    print("Beginning HQ Wav2Lip background training loop...")
    
    # Run the original train loop
    train(
        device, model, disc, train_data_loader, test_data_loader, optimizer, disc_optimizer,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=hparams.checkpoint_interval,
        nepochs=hparams.nepochs
    )