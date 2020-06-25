import math
import re
import glob
import os
from collections import defaultdict

import numpy as np
import scipy
import librosa
import matplotlib.pyplot as plt

from utils import endpoint_detect

RAW_PATHS = ['raw_datasets/**/*.wav', 'raw_datasets/**/*.dat']
DATASET_DIR = 'datasets'

SAMPLE_RATE = 8000
DURATION = 2
NFRAMES = DURATION * SAMPLE_RATE

DROP_BEGIN_DURATION = 0.2

NCLASSES = 20

classname = ['数字', '语音', '识别', '上海', '北京',
             '考试', '课程', '可测', '科创', '客车',
             'Digital', 'Speech', 'Voice', 'Shanghai', 'Beijing',
             'China', 'Course', 'Test', 'Coding', 'Code']

assert len(classname) == NCLASSES

person2datasets = defaultdict(dict)
npeople = 0


def parse_person_and_word(path):
    fn = path.split('/')[-1].split('.')[-2]
    assert re.fullmatch(r"\d{11}[-_]\d{2}[-_]\d{2}", fn) is not None
    match = re.fullmatch(r"(\d{11})[-_](\d{2})[-_]\d{2}", fn)
    person = match.group(1)
    word = int(match.group(2))
    assert 0 <= word
    return person, word


def parse1(path):
    try:
        person, word = parse_person_and_word(path)

        if word not in person2datasets[person]:
            person2datasets[person][word] = []

        person2datasets[person][word].append(path)
    except Exception as e:
        print(path)
        raise e


def fix_data(x, sr, duration):
    """Padding or trimming data points to given duration"""
    target_nframes = int(sr * duration)
    nframes = len(x)
    assert target_nframes > 0
    d = abs(target_nframes - nframes)
    if nframes <= target_nframes:
        xx = np.pad(x, (d//2, d - d//2))
        return xx
    else:
        if d//2 == 0:
            return x[d-d//2:]
        else:
            return x[d-d//2:-(d//2)]


def process1(x, y, sr, plot=False):

    sample_len = len(x)

    # Fix keyboard click noise of 16307130362 and part of 16307130222
    x = x[int(sr*DROP_BEGIN_DURATION):]

    nperseg = int(0.03 * sr)
    nperseg = pow(2, math.ceil(math.log(nperseg)/math.log(2)))

    if plot:
        plt.figure()

    si, ei = endpoint_detect(x, nperseg=nperseg, noverlap=nperseg//3,
                             sample_rate=sr, plot=plot)
    x = x[si: ei+1]
    if len(x) < sample_len:
        x = np.pad(x, (0, sample_len - len(x)))

    assert len(x) == sample_len

    if plot:
        print(path, classname[y], len(x))
        plt.show()

    return x


def write_wav(path, x, sr):
    scipy.io.wavfile.write(path, sr, x)


if __name__ == '__main__':

    # Parse
    for path in RAW_PATHS:
        for fn in glob.iglob(path, recursive=True):
            parse1(fn)

    # Test and Transform
    for person, word2paths in person2datasets.items():
        assert len(word2paths) == 20
        print(f'processing {person}', end='', flush=True)
        for word, paths in word2paths.items():
            assert len(paths) == 20
            for i, path in enumerate(paths):
                x, sr = librosa.load(path, sr=SAMPLE_RATE)
                x = fix_data(x, sr, DURATION)
                # x = process1(x, path2class(path), sr)
                assert len(x) == int(DURATION * sr)

                new_path = f'{DATASET_DIR}/{person}/{person}' + \
                           f'-{word:02d}-{i:02d}.wav'

                os.makedirs(os.path.dirname(new_path), exist_ok=True)
                write_wav(new_path, x, sr)
            print('.', end='', flush=True)
        print(f'finished!')

    print(f"number of people: {len(person2datasets)}",
          sorted(list(person2datasets.keys())))
