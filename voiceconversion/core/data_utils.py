import os
import random
import torch
import torch.utils.data
import numpy as np
import soundfile as sf

from core import commons
from core.mel_processing import spectrogram_torch


class TextAudioLoaderMultiNSFsid(torch.utils.data.Dataset):
    """
    Dataset loader for RVC training.
    Loads the preprocessed audio, semantic features (HuBERT), and pitch data.
    """
    def __init__(self, audiopaths_and_text, hparams):
        self.audiopaths_and_text = self.load_filepaths_and_text(audiopaths_and_text)
        self.hparams = hparams
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.win_length = hparams.win_length

        # Used for bucketing to group similar length sequences together
        self.lengths = []
        for row in self.audiopaths_and_text:
            feat_path = row[1]
            try:
                # Estimate length based on feature file size to avoid loading everything into memory
                size = os.path.getsize(feat_path) // (256 * 4) 
                self.lengths.append(size)
            except:
                self.lengths.append(100)
                
        random.seed(hparams.seed)
        random.shuffle(self.audiopaths_and_text)

    def get_audio_text_pair(self, audiopath_and_text):
        file_path, feature_path, f0_path, f0nsf_path, sid = audiopath_and_text
        
        # 1. Load audio
        audio, sr = sf.read(file_path)
        audio = torch.FloatTensor(audio)
        audio_norm = audio.unsqueeze(0)
        
        # 2. Compute Spectrogram
        spec = spectrogram_torch(
            audio_norm, 
            self.filter_length, 
            self.sampling_rate, 
            self.hop_length, 
            self.win_length, 
            center=False
        )
        spec = torch.squeeze(spec, 0)
        
        # 3. Load Semantic Features (HuBERT)
        phone = np.load(feature_path)
        phone = np.repeat(phone, 2, axis=0)  # Upsample resolution to match
        phone = torch.FloatTensor(phone)
        
        # 4. Load Pitch Extracts
        pitch = np.load(f0_path)
        pitchf = np.load(f0nsf_path)
        pitch = torch.LongTensor(pitch)
        pitchf = torch.FloatTensor(pitchf)
        
        sid = torch.LongTensor([int(sid)])
        
        return (phone, pitch, pitchf, spec, audio_norm, sid)

    def __getitem__(self, index):
        return self.get_audio_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)

    def load_filepaths_and_text(self, filename):
        with open(filename, encoding='utf-8') as f:
            filepaths_and_text = [line.strip().split("|") for line in f]
        return filepaths_and_text


class TextAudioCollateMultiNSFsid:
    """
    Zero-pads model inputs to the maximum length in the batch.
    """
    def __call__(self, batch):
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[3].size(1) for x in batch]),
            dim=0, descending=True)

        max_phone_len = max([x[0].size(0) for x in batch])
        max_spec_len = max([x[3].size(1) for x in batch])
        max_wav_len = max([x[4].size(1) for x in batch])

        phone_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))
        
        phone_padded = torch.FloatTensor(len(batch), max_phone_len, batch[0][0].shape[1])
        pitch_padded = torch.LongTensor(len(batch), max_phone_len)
        pitchf_padded = torch.FloatTensor(len(batch), max_phone_len)
        spec_padded = torch.FloatTensor(len(batch), batch[0][3].size(0), max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        sid = torch.LongTensor(len(batch))

        phone_padded.zero_()
        spec_padded.zero_()
        wav_padded.zero_()
        pitch_padded.zero_()
        pitchf_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            phone = row[0]
            phone_padded[i, :phone.size(0), :] = phone
            phone_lengths[i] = phone.size(0)

            pitch = row[1]
            pitch_padded[i, :pitch.size(0)] = pitch

            pitchf = row[2]
            pitchf_padded[i, :pitchf.size(0)] = pitchf

            spec = row[3]
            spec_padded[i, :, :spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = row[4]
            wav_padded[i, :, :wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

            sid[i] = row[5]

        return phone_padded, phone_lengths, pitch_padded, pitchf_padded, spec_padded, spec_lengths, wav_padded, wav_lengths, sid


class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Maintains buckets of similar sized data to minimize the amount of padding 
    needed during collation, significantly speeding up training.
    """
    def __init__(self, dataset, batch_size, boundaries, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries
        
        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum([t - t % self.batch_size for t in self.num_samples_per_bucket])
        self.num_samples = self.total_size // self.num_replicas

    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) + 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)

        for i in range(len(buckets) - 1, 0, -1):
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i-1)

        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (total_batch_size - (len_bucket % total_batch_size)) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []
        if self.shuffle:
            for bucket in self.buckets:
                indices.append(torch.randperm(len(bucket), generator=g).tolist())
        else:
            for bucket in self.buckets:
                indices.append(list(range(len(bucket))))

        batches = []
        for i in range(len(self.buckets)):
            bucket = self.buckets[i]
            len_bucket = len(bucket)
            ids_bucket = indices[i]
            num_samples_bucket = self.num_samples_per_bucket[i]

            rem = num_samples_bucket - len_bucket
            ids_bucket = ids_bucket + ids_bucket * (rem // len_bucket) + ids_bucket[:(rem % len_bucket)]

            ids_bucket = ids_bucket[self.rank::self.num_replicas]

            for j in range(len(ids_bucket) // self.batch_size):
                batch = [bucket[idx] for idx in ids_bucket[j*self.batch_size:(j+1)*self.batch_size]]
                batches.append(batch)

        if self.shuffle:
            batch_ids = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in batch_ids]
        self.batches = batches

        assert len(self.batches) * self.batch_size == self.num_samples
        return iter(self.batches)

    def _bisect(self, x, lo=0, hi=None):
        if lo < 0:
            raise ValueError('lo must be non-negative')
        if hi is None:
            hi = len(self.boundaries)
        while lo < hi:
            mid = (lo+hi)//2
            if self.boundaries[mid] < x:
                lo = mid+1
            else:
                hi = mid
        return lo

    def __len__(self):
        return self.num_samples // self.batch_size