import glob
import re
import os
from collections import defaultdict

import numpy as np
import scipy
import librosa

RAW_PATHS = ['raw_datasets/**/*.wav', 'raw_datasets/**/*.dat']
DATASET_DIR = 'datasets'

SAMPLE_RATE = 8000
DURATION = 2
NFRAMES = DURATION * SAMPLE_RATE

NCLASSES = 20

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
        for word, paths in word2paths.items():
            assert len(paths) == 20
            for i, path in enumerate(paths):
                x, sr = librosa.load(path, sr=SAMPLE_RATE)
                x = fix_data(x, sr, DURATION)
                assert len(x) == int(DURATION * sr)

                new_path = f'{DATASET_DIR}/{person}/{person}' + \
                           f'-{word:02d}-{i:02d}.wav'

                os.makedirs(os.path.dirname(new_path), exist_ok=True)
                write_wav(new_path, x, sr)

    print(f"number of people: {len(person2datasets)}",
          sorted(list(person2datasets.keys())))
