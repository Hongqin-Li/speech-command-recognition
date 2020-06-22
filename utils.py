from functools import lru_cache

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct


def dft(xs):
    """Vanilla DFT implementation"""
    xs = np.asarray(xs, dtype=float)
    N = xs.shape[0]
    ns = np.arange(N)
    k = ns.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * ns / N)

    return np.dot(M, xs)


def fft(xs):
    """Recursive implementation of the 1D Cooley-Tukey FFT"""
    xs = np.asarray(xs, dtype=float)
    N = xs.shape[0]

    assert N % 2 == 0

    if N <= 4:
        return dft(xs)
    else:
        x_even, x_odd = fft(xs[::2]), fft(xs[1::2])
        factor = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([x_even + factor[:N // 2] * x_odd,
                               x_even + factor[N // 2:] * x_odd])


def hamming(m):
    n = np.arange(0, m)
    return 0.54 - 0.46*np.cos(2*np.pi*n/(m-1))


def data2frames(x, nperseg, noverlap):
    # print(nperseg, noverlap)
    n = len(x)
    return np.asarray([x[i: i+nperseg] for i in range(0, n, nperseg-noverlap)
                       if i + nperseg <= n])


def stft(x, nperseg=256, noverlap=None):
    """FFT-based Short-time Fourier transform with hamming window"""
    n = x.size
    win = hamming(nperseg + 1)[:-1]
    if noverlap is None:
        noverlap = nperseg // 2

    return np.asarray([fft(x[i: i + nperseg] * win)[:nperseg//2+1]
                       for i in range(0, n, nperseg-noverlap)
                       if i + nperseg <= n]).T


def short_time_energy(*args, **kwargs):
    """Calculate the Short Time Energy of given frame"""
    frames = data2frames(*args, **kwargs)
    return np.sum(frames**2, axis=-1)


def zero_crossing_rate(*args, **kwargs):
    """Calculate zero-corssing rate of a given frame"""
    frames = data2frames(*args, **kwargs)
    return np.array([np.sum([1 for x in frame[1:] * frame[:-1] if x < 0])
                    / (len(frame)-1) for frame in frames])


def endpoint_detect(x, sample_rate, nperseg, noverlap, output_all=False,
                    plot=False):
    hop_length = nperseg - noverlap

    energy = short_time_energy(x, nperseg=nperseg, noverlap=noverlap)
    zcr = zero_crossing_rate(x, nperseg=nperseg, noverlap=noverlap)

    assert len(energy) == len(zcr)

    slient_end = 5

    mh = np.average(energy) / 4
    ml = (mh + np.average(energy[:slient_end])) / 4
    z0 = np.average(zcr[:slient_end])

    n1, n2 = 0, len(energy)-1
    starts, ends = [], []
    step = int(0.05*sample_rate/hop_length)

    # print(f'MH: {mh}, ML: {ml}, z0: {z0}, \
    #         step: {step}({step*hop_length/sample_rate}s)')

    while n1 + step < len(energy):
        if energy[n1] >= mh:
            break
        else:
            n1 += step
    while n2 - step >= 0:
        if energy[n2] >= mh:
            break
        else:
            n2 -= step

    starts.append(n1)
    ends.append(n2)

    while n1 - step >= 0:
        if energy[n1] < ml:
            break
        else:
            n1 -= step
    while n2 + step < len(energy):
        if energy[n2] < ml:
            break
        else:
            n2 += step

    starts.append(n1)
    ends.append(n2)

    maxd = 0.25 * sample_rate / hop_length
    d = 0
    while n1 - step >= 0:
        if zcr[n1] <= 3*z0 or d > maxd:
            break
        else:
            n1 -= step
            d += step

    d = 0
    while n2 + step < len(zcr):
        if zcr[n2] <= 3*z0 or d > maxd:
            break
        else:
            n2 += step
            d += step

    starts.append(n1)
    ends.append(n2)

    assert 0 <= n1 < len(energy) and 0 <= n2 < len(energy)
    assert len(starts) == len(ends) == 3

    if plot:
        time_per_frame = 1 / sample_rate
        startt = np.array(starts) * hop_length * time_per_frame
        endt = np.array(ends) * hop_length * time_per_frame
        minx, maxx = min(x), max(x)

        plt.subplot(311)
        waveplot(x, sample_rate=sample_rate)
        plt.plot([startt[0], startt[0]], [minx, maxx], c='b')
        plt.plot([startt[1], startt[1]], [minx, maxx], c='g')
        plt.plot([startt[2], startt[2]], [minx, maxx], c='r')
        plt.plot([endt[0], endt[0]], [minx, maxx], c='b')
        plt.plot([endt[1], endt[1]], [minx, maxx], c='g')
        plt.plot([endt[2], endt[2]], [minx, maxx], c='r')

        plt.subplot(312)
        waveplot(energy, sample_rate=sample_rate/(nperseg//2),
                 ylabel='short time energy')

        plt.subplot(313)
        waveplot(zcr, sample_rate=sample_rate/(nperseg//2),
                 ylabel='zero crossing rate')

    if output_all:
        return np.array(starts)*hop_length, np.array(ends)*hop_length, \
               energy, zcr
    else:
        return n1*hop_length, n2*hop_length


def power2db(x, amin=1e-10, top_db=80):
    magnitude = np.abs(x)

    db = 10.0 * np.log10(np.maximum(amin, magnitude))
    if top_db is not None:
        assert top_db > 0
        db = np.maximum(db, db.max() - top_db)
    return db


def db2power(x):
    return np.power(10.0, 0.1 * x)


def mel2hz(mels):
    mels = np.asarray(mels)
    return 700 * (10**(mels / 2595) - 1)


def hz2mel(freqs):
    freqs = np.asarray(freqs)
    return 2595 * np.log10(1 + freqs / 700)


@lru_cache()
def mel(sr, nperseg, nmels=128, fmin=0., fmax=None, norm='slaney'):
    """
    Create a Filterbank matrix to combine FFT bins into
    Mel-frequency bins
    """

    if fmax is None:
        fmax = sr / 2

    n_fft = nperseg

    # Center freqs of each FFT bin
    fft_freqs = np.linspace(0, sr / 2, 1 + n_fft//2, endpoint=True)
    # 'Center freqs' of mel bands - uniformly spaced between limits
    mel_freqs = mel2hz(np.linspace(hz2mel(fmin), hz2mel(fmax), nmels + 2))

    weights = np.zeros((nmels, 1 + n_fft//2))

    fdiff = np.diff(mel_freqs)
    ramps = np.subtract.outer(mel_freqs, fft_freqs)

    for i in range(nmels):
        # lower and upper slopes for all bins
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i+2] / fdiff[i+1]

        # .. then intersect them with each other and zero
        weights[i] = np.maximum(0, np.minimum(lower, upper))

    if norm == 'slaney':
        # Slaney-style mel is scaled to be approx constant energy per channel
        enorm = 2.0 / (mel_freqs[2:nmels+2] - mel_freqs[:nmels])
        weights *= enorm[:, np.newaxis]

    return weights


def mfcc(x, sample_rate, nperseg, noverlap=None, nmfcc=20, power=2, **kwargs):
    spect = np.abs(stft(x, nperseg=nperseg, noverlap=noverlap))**power

    mel_basis = mel(sr=sample_rate, nperseg=nperseg, **kwargs)
    mel_spect = np.dot(mel_basis, spect)

    return dct(power2db(mel_spect), axis=0, type=2, norm='ortho')[:nmfcc]


def normalize(x, axis=0):
    return x / np.max(np.abs(x), axis=axis, keepdims=True)


def pre_emphasis(x, coef=0.97):
    return x - coef * np.append(1, x[:-1])


def waveplot(y, sample_rate, xlabel="time(s)", ylabel="amplitude"):
    nframes = len(y)
    time = np.arange(0, nframes) / sample_rate
    plt.plot(time, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
