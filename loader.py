import glob
import re
import random
import math

import librosa
import torch
from torch.utils.data import Dataset, DataLoader
from preprocess import DATASET_DIR, SAMPLE_RATE, NFRAMES, NCLASSES
from utils import pre_emphasis, normalize
# from utils import normalize, mfcc

BATCH_SIZE = 64
WINDOW_DURATION = 0.03
NMFCC = 64

NPERSEG = pow(2, math.ceil(math.log(int(WINDOW_DURATION * SAMPLE_RATE))
                           / math.log(2)))


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

        x = pre_emphasis(normalize(x))
        x = librosa.feature.mfcc(x, sr=sr, n_fft=NPERSEG,
                                 hop_length=2*NPERSEG//3, n_mfcc=NMFCC)

        # x = mfcc(x, sample_rate=sr, nperseg=NPERSEG, noverlap=NPERSEG//3,
        #          nmfcc=NMFCC)

        x = torch.from_numpy(x)
        y = torch.LongTensor([self.path2class(self.paths[idx])])
        return x, y

    def __len__(self):
        return len(self.paths)


def get_dataloaders(ntest=9, kfold=10, batch_size=BATCH_SIZE):
    '''
    Args:
        ntest: ntest people will be used as test set.
        kfold: 1/kfold of the remaining samples are used as
               validation set, else as trainning set.
    '''

    people = [path.split('/')[-1] for path in glob.glob(f'{DATASET_DIR}/*')]
    # random.shuffle(people)
    people.sort()
    traindev_people, test_people = people[:-ntest], people[-ntest:]

    traindev_paths = get_paths(traindev_people)
    random.shuffle(traindev_paths)
    dev_samples = len(traindev_paths) // kfold

    train_dataset = AudioDataset(traindev_paths[:-dev_samples], path2class)
    dev_dataset = AudioDataset(traindev_paths[-dev_samples:], path2class)
    test_dataset = AudioDataset(get_paths(test_people), path2class)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=True)

    print(f'#train: {len(train_loader.dataset)}, ' +
          f'#dev: {len(dev_loader.dataset)}, ' +
          f'#test: {len(test_loader.dataset)}')
    print(f'#classes: {NCLASSES}')
    print(f'Batch size: {BATCH_SIZE}')
    print(f'people: {people}')
    print(f'test people: {test_people}')

    return train_loader, dev_loader, test_loader


if __name__ == '__main__':
    get_dataloaders()
