import os

class HParams:
    def __init__(self, **kwargs):
        self.data = {}
        for key, value in kwargs.items():
            self.data[key] = value

    def __getattr__(self, key):
        if key not in self.data:
            raise AttributeError(f"'HParams' object has no attribute {key}")
        return self.data[key]

    def set_hparam(self, key, value):
        self.data[key] = value

# Default hyperparameters for the API Microservice
hparams = HParams(
    num_mels=80, 
    rescale=True,  
    rescaling_max=0.9,  
    use_lws=False,
    n_fft=800,  
    hop_size=200,  
    win_size=800,  
    sample_rate=16000,  
    frame_shift_ms=None,  
    signal_normalization=True,
    allow_clipping_in_normalization=True,  
    symmetric_mels=True,
    max_abs_value=4.,
    preemphasize=True,  
    preemphasis=0.97,  
    min_level_db=-100,
    ref_level_db=20,
    fmin=55,
    fmax=7600,  
    img_size=96,
    fps=25,
    batch_size=16,
    initial_learning_rate=1e-4,
    nepochs=200000000000000000, 
    num_workers=16,
    checkpoint_interval=3000,
    eval_interval=3000,
    save_optimizer_state=True,
    syncnet_wt=0.0, 
    syncnet_batch_size=64,
    syncnet_lr=1e-4,
    syncnet_eval_interval=10000,
    syncnet_checkpoint_interval=10000,
    disc_wt=0.07,
    disc_initial_learning_rate=1e-4,
)