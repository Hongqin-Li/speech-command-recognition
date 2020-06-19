import numpy as np
import librosa
from scipy import signal
from utils import fft, hamming, stft, \
                  zero_crossing_rate, \
                  power2db, db2power, normalize, \
                  mel2hz, hz2mel, mel, mfcc, pre_emphasis


def test_fft():
    x = np.random.random(512)
    assert np.allclose(fft(x), np.fft.fft(x))


def test_hamming():
    m = 10
    assert np.allclose(hamming(m), np.hamming(m))


def test_stft_of_scipy_librosa():
    nperseg = 128
    x = np.random.random(nperseg*4)
    a = signal.stft(x, window='hamming', nperseg=nperseg,
                    boundary=None, padded=False)[-1]
    b = librosa.core.stft(x, window='hamming', n_fft=nperseg,
                          hop_length=nperseg//2,
                          center=False)/(nperseg//2)
    assert np.allclose(a, b, rtol=0.1)


def test_stft():
    nperseg = 4
    noverlap = nperseg // 2
    # x = np.array([1., 2., 3., 4., 5., 6.])
    x = np.random.random(nperseg*4)

    a = librosa.core.stft(x, window='hamming',
                          n_fft=nperseg, hop_length=nperseg-noverlap,
                          center=False)
    b = stft(x, nperseg=nperseg, noverlap=noverlap)
    assert np.allclose(a, b)


def test_zero_crossing_rate():
    nperseg = 128
    noverlap = nperseg // 2
    x = np.random.random(nperseg*4)
    a = librosa.feature.zero_crossing_rate(x, frame_length=nperseg,
                                           hop_length=nperseg-noverlap,
                                           center=False)
    b = zero_crossing_rate(x, nperseg=nperseg, noverlap=noverlap)
    assert np.allclose(a, b)


def test_power2db():
    x = np.random.random(10)*1000
    assert np.allclose(power2db(x), librosa.power_to_db(x))


def test_db2power():
    x = np.random.random(10)*40
    assert np.allclose(db2power(x), librosa.db_to_power(x))


def test_mel2hz():
    m = np.random.random(10)
    assert np.allclose(mel2hz(m), librosa.mel_to_hz(m, htk=True))


def test_hz2mel():
    h = np.random.random(10)
    assert np.allclose(hz2mel(h), librosa.hz_to_mel(h, htk=True))


def test_mel():
    sample_rate, nperseg, nmels, fmax = 8000, 1024, 128, 4000
    a = mel(sr=sample_rate, nperseg=nperseg, nmels=nmels, fmax=fmax)
    b = librosa.filters.mel(sr=sample_rate, n_fft=nperseg, n_mels=nmels,
                            fmax=fmax, htk=True)
    assert a.shape == b.shape
    assert np.allclose(a, b)


def test_mfcc():
    data = np.arange(1024, dtype=np.float32)
    data = data / np.max(np.abs(data))
    sr, nmfcc, nmels, nperseg, noverlap = 8000, 20, 64, 256, 128
    a = mfcc(data, sample_rate=sr, nmfcc=nmfcc, nperseg=nperseg,
             noverlap=noverlap, nmels=nmels)
    b = librosa.feature.mfcc(data, sr=sr, n_mfcc=nmfcc, n_mels=nmels,
                             n_fft=nperseg, hop_length=nperseg-noverlap,
                             window='hamming', center=False, htk=True)
    assert a.shape == b.shape
    assert np.allclose(a, b)


def test_normalize():
    x = np.array([-10, 0, 1.])
    assert np.allclose(normalize(x), librosa.util.normalize(x))

    x = np.array([-2, 0, 10.])
    assert np.allclose(normalize(x), librosa.util.normalize(x))

    x = np.array([[-1, 2], [-2, 1.]])
    assert np.allclose(normalize(x), librosa.util.normalize(x))


def test_pre_emphasis():
    x = np.random.random(10)
    assert np.allclose(pre_emphasis(x, coef=0.9),
                       librosa.effects.preemphasis(x, coef=0.9))
