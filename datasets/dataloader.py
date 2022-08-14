import glob
import os
import random
from pathlib import Path

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from utils.utils import read_mel_np, read_wav_np


def create_dataloader(hp, args, train):
    dataset = MelFromDisk(hp, args, train)

    if train:
        return DataLoader(
            dataset=dataset,
            batch_size=hp.train.batch_size,
            shuffle=True,
            num_workers=hp.train.num_workers,
            pin_memory=True,
            drop_last=True,
        )
    else:
        return DataLoader(
            dataset=dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
        )


class MelFromDisk(Dataset):
    def __init__(self, hp, args, train):
        self.hp = hp
        self.args = args
        self.train = train
        self.wav_list = list(
            map(
                Path,
                Path(hp.data.train if train else hp.data.validation)
                .read_text()
                .splitlines(),
            )
        )
        self.mel_list = list(
            map(
                Path,
                Path(hp.data.train_mel if train else hp.data.validation_mel)
                .read_text()
                .splitlines(),
            )
        )
        # print("Wavs path :", self.path)
        # print(self.hp.data.mel_path)
        # print("Length of wavelist :", len(self.wav_list))
        self.mel_segment_length = hp.audio.segment_length // hp.audio.hop_length + 2
        self.mapping = [i for i in range(len(self.wav_list))]

    def __len__(self):
        return len(self.wav_list)

    def __getitem__(self, idx):

        if self.train:
            idx1 = idx
            idx2 = self.mapping[idx1]
            return self.my_getitem(idx1), self.my_getitem(idx2)
        else:
            return self.my_getitem(idx)

    def shuffle_mapping(self):
        random.shuffle(self.mapping)

    def my_getitem(self, idx):
        wavpath = self.wav_list[idx]
        mel_path = self.mel_list[idx]

        sr, audio = read_wav_np(wavpath, self.hp.audio.sampling_rate)
        audio_padsize = (
            self.hp.audio.segment_length + self.hp.audio.pad_short - len(audio)
        )
        if audio_padsize > 0:
            audio = np.pad(
                audio, (0, audio_padsize), mode="constant", constant_values=0.0
            )

        audio = torch.from_numpy(audio).unsqueeze(0)
        # mel = torch.load(melpath).squeeze(0) # # [num_mel, T]

        mel = read_mel_np(mel_path)
        if audio_padsize > 0:
            mel = np.pad(
                mel,
                ((0, 0), (0, audio_padsize // self.hp.audio.hop_length)),
                mode="constant",
                constant_values=1e-5,
            )
        mel = torch.from_numpy(mel)

        if self.train:
            max_mel_start = mel.size(1) - self.mel_segment_length
            mel_start = random.randint(0, max_mel_start)
            mel_end = mel_start + self.mel_segment_length
            mel = mel[:, mel_start:mel_end]

            audio_start = mel_start * self.hp.audio.hop_length
            audio = audio[:, audio_start : audio_start + self.hp.audio.segment_length]
            assert audio.shape[1] >= self.hp.audio.segment_length, wavpath

        audio = audio + (1 / 32768) * torch.randn_like(audio)
        return mel, audio


def collate_fn(batch):

    sr = 22050
    # perform padding and conversion to tensor
    mels_g = [x[0][0] for x in batch]
    audio_g = [x[0][1] for x in batch]

    mels_g = torch.stack(mels_g)
    audio_g = torch.stack(audio_g)

    sub_orig_1 = torchaudio.transforms.Resample(sr, (sr // 2))(audio_g)
    sub_orig_2 = torchaudio.transforms.Resample(sr, (sr // 4))(audio_g)
    sub_orig_3 = torchaudio.transforms.Resample(sr, (sr // 8))(audio_g)
    sub_orig_4 = torchaudio.transforms.Resample(sr, (sr // 16))(audio_g)

    mels_d = [x[1][0] for x in batch]
    audio_d = [x[1][1] for x in batch]
    mels_d = torch.stack(mels_d)
    audio_d = torch.stack(audio_d)
    sub_orig_1_d = torchaudio.transforms.Resample(sr, (sr // 2))(audio_d)
    sub_orig_2_d = torchaudio.transforms.Resample(sr, (sr // 4))(audio_d)
    sub_orig_3_d = torchaudio.transforms.Resample(sr, (sr // 8))(audio_d)
    sub_orig_4_d = torchaudio.transforms.Resample(sr, (sr // 16))(audio_d)

    return [mels_g, audio_g, sub_orig_1, sub_orig_2, sub_orig_3, sub_orig_4], [
        mels_d,
        audio_d,
        sub_orig_1_d,
        sub_orig_2_d,
        sub_orig_3_d,
        sub_orig_4_d,
    ]
