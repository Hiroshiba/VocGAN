import random
import subprocess
from pathlib import Path

import numpy as np
from librosa.core import load


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_commit_hash():
    message = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
    return message.strip().decode("utf-8")


def read_wav_np(path: Path, sampling_rate=None):
    if path.suffix != ".npy":
        data, sampling_rate = load(path, sr=sampling_rate)
    else:
        a = np.load(path, allow_pickle=True).item()
        assert sampling_rate == a["rate"]
        data = a["array"]
    return sampling_rate, data


def read_mel_np(path: Path):
    a = np.load(path, allow_pickle=True).item()
    data = a["array"].T
    return data
