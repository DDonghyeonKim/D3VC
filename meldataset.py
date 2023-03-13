#coding: utf-8

import os
import time
import random
import random

import librosa
import torch
import torchaudio

import numpy as np
import soundfile as sf
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from audiomentations import TimeStretch
import scipy.signal as ss
import soundfile as sf
import rir_generator as rir
import logging

#torch.multiprocessing.set_start_method('spawn')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

np.random.seed(1)
random.seed(1)

SPECT_PARAMS = {
    "n_fft": 2048,
    "win_length": 1200,
    "hop_length": 300
}
MEL_PARAMS = {
    "n_mels": 80,
    "n_fft": 2048,
    "win_length": 1200,
    "hop_length": 300,
}

class MelDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_list,
                 sr=24000,
                 validation=False,
                 ):

        _data_list = [l[:-1].split('|') for l in data_list]
        self.data_list = [(path, int(label)) for path, label in _data_list]
        self.data_list_per_class = {
            target: [(path, label) for path, label in self.data_list if label == target] \
            for target in list(set([label for _, label in self.data_list]))}

        self.sr = sr
        self.to_melspec = torchaudio.transforms.MelSpectrogram(**MEL_PARAMS)

        self.mean, self.std = -4, 4
        self.validation = validation
        self.max_mel_length = 192

        #wav2vec.extract_features(mc_src, None)
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]

        mel_tensor, label,w2v = self._load_data(data,True)
        ref_data = random.choice(self.data_list)
        ref_mel_tensor, ref_label,_ = self._load_data(ref_data,True)
        ref2_data = random.choice(self.data_list_per_class[ref_label])
        ref2_mel_tensor, _,_ = self._load_data(ref2_data,True)
        return mel_tensor, label, ref_mel_tensor, ref2_mel_tensor, ref_label,w2v

    def _load_data(self, path,Aug=False):
        wave_tensor, label,audios = self._load_tensor(path,Aug)
        #print(audios.shape)

        if not self.validation: # random scale for robustness
                    ###RIR

            random_scale = 0.5 + 0.5 * np.random.random()
            wave_tensor = random_scale * wave_tensor
            mel_tensor = self.to_melspec(wave_tensor)
#            mel_tensor2 = self.to_melspec(wave_tensor)##192
            mel_tensor = (torch.log(1e-5 + mel_tensor) - self.mean) / self.std
            M_size=mel_tensor.size()
            if Aug:
                Thr = np.random.rand(1)
                if Thr>0.5:
                    TM=torchaudio.transforms.TimeMasking(time_mask_param=10).to('cuda:0')
                    FM=torchaudio.transforms.FrequencyMasking(freq_mask_param=5).to('cuda:0')
                    mel_tensor=FM(TM(mel_tensor.unsqueeze(0))).squeeze(0)
        return mel_tensor,label,audios

    def _preprocess(self, wave_tensor, ):
        mel_tensor = self.to_melspec(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor) - self.mean) / self.std
        return mel_tensor

    def _load_tensor(self, data,Aug=False):
        wave_path, label = data
        label = int(label)
        wave, sr = sf.read(wave_path.replace('\\','/'))
        if sr != 24000:
            wave=librosa.resample(wave,16000,24000)
        if Aug:
            Thr=np.random.rand(1)
            if  Thr< 0.20:
                transform = TimeStretch(
                    min_rate=0.8,
                    max_rate=1.25,
                    leave_length_unchanged=True,
                    p=1.0
                )
                wave = transform(wave, sample_rate=sr)
            elif Thr< 0.40:
                h = rir.generate(
                    c=340,  # Sound velocity (m/s)
                    fs=sr,  # Sample frequency (samples/s)
                    r=[  # Receiver position(s) [x y z] (m)
                        [2, 1.5, 1]
                    ],
                    s=[2, 3.5, 2],  # Source position [x y z] (m)
                    L=[5, 4, 6],  # Room dimensions [x y z] (m)
                    reverberation_time=0.4,  # Reverberation time (s)
                    nsample=4096,  # Number of output samples
                )
                wave = ss.convolve(wave,  np.squeeze(h,1),mode='same')
            elif Thr < 0.50:

                transform = TimeStretch(
                    min_rate=0.8,
                    max_rate=1.25,
                    leave_length_unchanged=True,
                    p=1.0
                )
                wave = transform(wave, sample_rate=sr)
                h = rir.generate(
                    c=340,  # Sound velocity (m/s)
                    fs=sr,  # Sample frequency (samples/s)
                    r=[  # Receiver position(s) [x y z] (m)
                        [2, 1.5, 1]
                    ],
                    s=[2, 3.5, 2],  # Source position [x y z] (m)
                    L=[5, 4, 6],  # Room dimensions [x y z] (m)
                    reverberation_time=0.4,  # Reverberation time (s)
                    nsample=4096,  # Number of output samples
                )
                wave = ss.convolve(wave,  np.squeeze(h,1),mode='same')
        thr=57500
        if len(wave)> thr:
            ind=np.arange(len(wave)-thr)
            np.random.shuffle(ind)
            wave=wave[ind[0]:ind[0]+thr]
        elif len(wave)< thr:
            wave = np.concatenate([wave,np.zeros([thr-len(wave)])],0)
        wave_tensor = torch.from_numpy(wave).float()
        wave2= librosa.resample(wave, 24000,16000)
#        print(len(wave2))
        wave_tensor2 = torch.from_numpy(wave2).float()

        return wave_tensor, label,wave_tensor2

class Collater(object):
    """
    Args:
      adaptive_batch_size (bool): if true, decrease batch size when long data comes.
    """

    def __init__(self, return_wave=False):
        self.text_pad_index = 0
        self.return_wave = return_wave
        self.max_mel_length = 192
        self.mel_length_step = 16
        self.latent_dim = 16

    def __call__(self, batch):
        batch_size = len(batch)
        nmels = batch[0][0].size(0)
        mels = torch.zeros((batch_size, nmels, self.max_mel_length)).float()
        labels = torch.zeros((batch_size)).long()
        ref_mels = torch.zeros((batch_size, nmels, self.max_mel_length)).float()
        ref2_mels = torch.zeros((batch_size, nmels, self.max_mel_length)).float()
        ref_labels = torch.zeros((batch_size)).long()
        WV_feature = torch.zeros((batch_size,38334)).float()
        for bid, (mel, label, ref_mel, ref2_mel, ref_label,WB) in enumerate(batch):
            mel_size = mel.size(1)
            mels[bid, :, :mel_size] = mel
            
            ref_mel_size = ref_mel.size(1)
            ref_mels[bid, :, :ref_mel_size] = ref_mel
            
            ref2_mel_size = ref2_mel.size(1)
            ref2_mels[bid, :, :ref2_mel_size] = ref2_mel
            
            labels[bid] = label
            ref_labels[bid] = ref_label
            WV_feature[bid,:]=WB
        z_trg = torch.randn(batch_size, self.latent_dim)
        z_trg2 = torch.randn(batch_size, self.latent_dim)

        mels, ref_mels, ref2_mels = mels.unsqueeze(1), ref_mels.unsqueeze(1), ref2_mels.unsqueeze(1)


        return mels, labels, ref_mels, ref2_mels, ref_labels, z_trg, z_trg2,WV_feature

def build_dataloader(path_list,
                     validation=False,
                     batch_size=4,
                     num_workers=1,
                     device='cpu',
                     collate_config={},
                     dataset_config={}):

    dataset = MelDataset(path_list, validation=validation)
    collate_fn = Collater(**collate_config)
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=(not validation),
                             num_workers=num_workers,
                             drop_last=True,
                             collate_fn=collate_fn,
                             pin_memory=(device != 'cpu'))

    return data_loader