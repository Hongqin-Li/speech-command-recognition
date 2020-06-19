import glob
import re
import random
from collections import Counter, defaultdict

import numpy as np
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
from preprocess import DATASET_DIR, SAMPLE_RATE, NFRAMES, NCLASSES
from utils import mfcc

BATCH_SIZE = 32
NPERSEG = 256

def path2class(path):
    fn = path.split('/')[-1].split('.')[-2]
    match = re.fullmatch(r"(\d{11})[-_](\d{2})[-_]\d{2}", fn)
    word = int(match.group(2))
    assert 0 <= word
    return word

def get_paths(people):
    r = [p for person in people for p in glob.glob(f'{DATASET_DIR}/{person}/*.wav')]
    return r

class AudioDataset(Dataset):
    def __init__(self, paths, path2class):
        self.paths = paths
        self.path2class = path2class

    def __getitem__(self, idx):
        x, sr = librosa.load(self.paths[idx], sr=None)
        assert sr == SAMPLE_RATE and len(x) == NFRAMES
        x = librosa.feature.mfcc(x, sr=sr, n_fft=NPERSEG, hop_length=NPERSEG//2, n_mfcc=64)
        # x = mfcc(x, sample_rate=sr, nperseg=NPERSEG)
        x = torch.from_numpy(x)
        x = torch.unsqueeze(x, 0)
        # Expand to 3 channels
        # x = x.expand(3, -1, -1)
        y = torch.LongTensor([self.path2class(self.paths[idx])])
        return x, y

    def __len__(self):
        return len(self.paths)

# Parse and test
people = [ path.split('/')[-1] for path in glob.glob(f'{DATASET_DIR}/*')]
npeople = len(people)

random.shuffle(people)

# FIXME tune test set size
ntest = 6
ntraindev = npeople - ntest
ndev = ntraindev // 5
ntrain = npeople - ntest - ndev
assert ntrain > 0 and ndev > 0 and ntest > 0

print(f'#train: {ntrain}, #dev: {ndev}, #test: {ntest}')
print(f'#classes: {NCLASSES}')
print(f'people: {people}')

train_dataset = AudioDataset(get_paths(people[:ntrain]), path2class)
dev_dataset = AudioDataset(get_paths(people[ntrain:ntrain+ndev]), path2class)
test_dataset = AudioDataset(get_paths(people[ntrain+ndev:]), path2class)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
