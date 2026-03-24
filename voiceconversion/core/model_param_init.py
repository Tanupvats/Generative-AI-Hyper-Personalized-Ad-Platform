import json

class ModelParameters(object):
    """
    Parses and stores hyperparameters for UVR5 (Ultimate Vocal Remover) models.
    These parameters define the STFT window sizes, hop lengths, and frequency bands
    used for the cascaded ASPP architecture.
    """
    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            self.param = json.load(f)

        # Ensure bands are indexed as integers for easier iteration in spec_utils
        if "band" in self.param:
            new_bands = {}
            for k, v in self.param["band"].items():
                new_bands[int(k)] = v
            self.param["band"] = new_bands
            
        # Standardize expected keys to prevent runtime errors if they are missing
        self._set_default_values()

    def _set_default_values(self):
        """Sets default values for optional UVR5 configuration keys."""
        defaults = {
            "sr": 44100,
            "n_fft": 2048,
            "hop_length": 512,
            "pre_filter_start": 0,
            "pre_filter_stop": 0,
            "aggressiveness": 1.0
        }
        for key, val in defaults.items():
            if key not in self.param:
                self.param[key] = val

    def get_band_params(self, band_id: int):
        """
        Returns specific STFT parameters for a given frequency band.
        UVR5 models often use different window sizes for low, mid, and high frequencies.
        """
        return self.param.get("band", {}).get(band_id)

    @property
    def sample_rate(self):
        return self.param.get("sr")

    @property
    def fft_size(self):
        return self.param.get("n_fft")