import glob
import re
import random

import librosa
import torch
from torch.utils.data import Dataset, DataLoader
from preprocess import DATASET_DIR, SAMPLE_RATE, NFRAMES, NCLASSES
# from utils import mfcc

BATCH_SIZE = 64
WINDOW_DURATION = 0.03
NMFCC = 128


def path2class(path):
    fn = path.split('/')[-1].split('.')[-2]
    match = re.fullmatch(r"(\d{11})[-_](\d{2})[-_]\d{2}", fn)
    word = int(match.group(2))
    assert 0 <= word
    return word


def get_paths(people):
    r = [p for person in people for p in
         glob.glob(f'{DATASET_DIR}/{person}/*.wav')]
    return r


class AudioDataset(Dataset):
    def __init__(self, paths, path2class):
        self.paths = paths
        self.path2class = path2class

    def __getitem__(self, idx):
        x, sr = librosa.load(self.paths[idx], sr=None)
        assert sr == SAMPLE_RATE and len(x) == NFRAMES
        nperseg = int(WINDOW_DURATION * SAMPLE_RATE)
        x = librosa.feature.mfcc(x, sr=sr, n_fft=nperseg,
                                 hop_length=2*nperseg//3, n_mfcc=NMFCC)
        # x = mfcc(x, sample_rate=sr, nperseg=nperseg,
        #          nmfcc=NMFCC).astype('float32')
        x = torch.from_numpy(x)
        y = torch.LongTensor([self.path2class(self.paths[idx])])
        return x, y

    def __len__(self):
        return len(self.paths)


# Parse and test
people = [path.split('/')[-1] for path in glob.glob(f'{DATASET_DIR}/*')]
npeople = len(people)

random.shuffle(people)

# FIXME tune test set size
ntest = 3

traindev_paths = get_paths(people[:-ntest])
random.shuffle(traindev_paths)
dev_samples = len(traindev_paths) // 10

train_dataset = AudioDataset(traindev_paths[:-dev_samples], path2class)
dev_dataset = AudioDataset(traindev_paths[-dev_samples:], path2class)
test_dataset = AudioDataset(get_paths(people[-ntest:]), path2class)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f'#train: {len(train_loader.dataset)}, ' +
      f'#dev: {len(dev_loader.dataset)}, ' +
      f'#test: {len(test_loader.dataset)}')
print(f'#classes: {NCLASSES}')
print(f'Batch size: {BATCH_SIZE}')
print(f'people: {people}')
