"""
@file base.py
Contains the functions that compute the features

calculate filterbank features. Provides e.g. fbank and mfcc features for use in
ASR applications

Author: James Lyons 2012

"""

import numpy
import sigproc
from scipy.fftpack import dct
from scipy.ndimage import convolve1d
import scipy.signal
import scipy


def raw(signal):
    """
    compute the raw audio signal with limited range

    Args:
        signal: the audio signal from which to compute features. Should be an
            N*1 array

    Returns:
        A numpy array of size (N by 1) containing the raw audio limited to a
        range between -1 and 1
    """
    feat = signal.astype(numpy.float32)/numpy.max(numpy.abs(signal))

    return feat[:, numpy.newaxis]


def spec(signal, samplerate, conf):
    """
    Compute complex spectrogram features from an audio signal.

    Args:
        signal: the audio signal from which to compute features. Should be an
            N*1 array
        samplerate: the samplerate of the signal we are working with.
        conf: feature configuration

    Returns:
        A numpy array of size (NUMFRAMES by numfreq) containing features. Each
        row holds 1 feature vector, a numpy vector containing the complex
        spectrum of the corresponding frame
    """
    raise BaseException('Not yet implemented')
    winfunc = _get_winfunc(conf['winfunc'])

    frames = sigproc.framesig(signal, float(conf['winlen'])*samplerate,
                              float(conf['winstep'])*samplerate,
                              winfunc)
    spec = sigproc.spec(frames, int(conf['nfft']))

    return spec


def spec2time(spec, samplerate, siglen, conf):
    """
    Compute the time domain signal from the complex spectrogram. No preemphasis is assumed.

    Args:
    spec: A numpy array of size (NUMFRAMES by numfreq) containing features. Each
        row holds 1 feature vector, a numpy vector containing the complex
        spectrum of the corresponding frame
        samplerate: the samplerate of the signal we are working with.
        siglen the: length of the desired signal, use 0 if unknown. Output will
            be truncated to siglen samples.
        conf: feature configuration

    Returns:
        signal: the audio signal from which to compute features. This is an
            N*1 array
    """

    raise BaseException('Not yet implemented')
    frames = sigproc.spec2frames(spec)

    winfunc = _get_winfunc(conf['winfunc'])

    signal = sigproc.deframesig(
        frames, siglen, float(conf['winlen'])*samplerate, float(conf['winstep'])*samplerate, winfunc)

    # Limit the range of the signal between -1.0 and 1.0
    signal = signal/numpy.max(numpy.abs(signal))

    return signal


def frames(signal, samplerate, conf):
    """
    Compute frames from an audio signal.

    Args:
        signal: the audio signal from which to compute features. Should be an
            N*1 array
        samplerate: the samplerate of the signal we are working with.
        conf: feature configuration

    Returns:
        A numpy array of size (NUMFRAMES by winlen) containing features. Each
        row holds 1 feature vector
    """
    raise BaseException('Not yet implemented')
    signal = sigproc.preemphasis(signal, float(conf['preemph']))

    winfunc = _get_winfunc(conf['winfunc'])

    frames = sigproc.framesig(signal, float(conf['winlen'])*samplerate,
                              float(conf['winstep'])*samplerate,
                              winfunc)

    return frames


def powspec(signal, samplerate, conf):
    """
    Compute squared magnitude spectrogram features from an audio signal.

    Args:
        signal: the audio signal from which to compute features. Should be an
            N*1 array
        samplerate: the samplerate of the signal we are working with.
        conf: feature configuration

    Returns:
        A numpy array of size (NUMFRAMES by numfreq) containing features. Each
        row holds 1 feature vector, a numpy vector containing the magnitude
        spectrum of the corresponding frame
    """

    raise BaseException('Not yet implemented')
    signal = sigproc.preemphasis(signal, float(conf['preemph']))

    winfunc = _get_winfunc(conf['winfunc'])

    frames = sigproc.framesig(signal, float(conf['winlen'])*samplerate,
                              float(conf['winstep'])*samplerate,
                              winfunc)
    powspec = sigproc.powspec(frames, int(conf['nfft']))

    return powspec


def magspec(signal, samplerate, conf):
    """
    Compute magnitude spectrogram features from an audio signal.

    Args:
        signal: the audio signal from which to compute features. Should be an
            N*1 array
        samplerate: the samplerate of the signal we are working with.
        conf: feature configuration

    Returns:
        A numpy array of size (NUMFRAMES by numfreq) containing features. Each
        row holds 1 feature vector, a numpy vector containing the magnitude
        spectrum of the corresponding frame
    """

    raise BaseException('Not yet implemented')
    signal = sigproc.preemphasis(signal, float(conf['preemph']))

    winfunc = _get_winfunc(conf['winfunc'])

    frames = sigproc.framesig(signal, float(conf['winlen'])*samplerate,
                              float(conf['winstep'])*samplerate,
                              winfunc)
    magspec = sigproc.magspec(frames, int(conf['nfft']))

    return magspec


def angspec(signal, samplerate, conf):
    """
    Compute angular spectrogram features from an audio signal.

    Args:
        signal: the audio signal from which to compute features. Should be an
            N*1 array
        samplerate: the samplerate of the signal we are working with.
        conf: feature configuration

    Returns:
        A numpy array of size (NUMFRAMES by numfreq) containing features. Each
        row holds 1 feature vector, a numpy vector containing the angular
        spectrum of the corresponding frame
    """

    raise BaseException('Not yet implemented')
    signal = sigproc.preemphasis(signal, float(conf['preemph']))

    winfunc = _get_winfunc(conf['winfunc'])

    frames = sigproc.framesig(signal, float(conf['winlen'])*samplerate,
                              float(conf['winstep'])*samplerate,
                              winfunc)
    angspec = sigproc.angspec(frames, int(conf['nfft']))

    return angspec


def logspec(signal, samplerate, conf):
    """
    Compute log magnitude spectrogram features from an audio signal.

    Args:
        signal: the audio signal from which to compute features. Should be an
            N*1 array
        samplerate: the samplerate of the signal we are working with.
        conf: feature configuration

    Returns:
        A numpy array of size (NUMFRAMES by numfreq) containing features. Each
        row holds 1 feature vector, a numpy vector containing the log magnitude
        spectrum of the corresponding frame
    """
    raise BaseException('Not yet implemented')
    signal = sigproc.preemphasis(signal, float(conf['preemph']))

    winfunc = _get_winfunc(conf['winfunc'])

    frames = sigproc.framesig(signal, float(conf['winlen'])*samplerate,
                              float(conf['winstep'])*samplerate,
                              winfunc)
    logspec = sigproc.logmagspec(frames, int(conf['nfft']))

    return logspec


def _get_winfunc(str_winfunc):
    """
    Get the requested window function.

    Args:
    str_winfunc: a string indicating the desired window function

    Returns:
    winfunc: the desired window function as a python lambda function
    """

    if str_winfunc == 'cosine':
        winfunc = scipy.signal.cosine
    elif str_winfunc == 'hanning':
        winfunc = scipy.hanning
    elif str_winfunc == 'hamming':
        winfunc = scipy.signal.hamming
    elif str_winfunc == 'none' or str_winfunc == 'None':
        winfunc = lambda x: numpy.ones((x, ))
    else:
        raise Exception('unknown window function: %s' % str_winfunc)

    return winfunc


def mfcc(signal, samplerate, conf):
    """
    Compute MFCC features from an audio signal.

    Args:
        signal: the audio signal from which to compute features. Should be an
            N*1 array
        samplerate: the samplerate of the signal we are working with.
        conf: feature configuration

    Returns:
        A numpy array of size (NUMFRAMES by numcep) containing features. Each
        row holds 1 feature vector, a numpy vector containing the signal
        log-energy
    """

    raise BaseException('Not yet implemented')
    feat, energy = fbank(signal, samplerate, conf)
    feat = numpy.log(feat)
    feat = dct(feat, type=2, axis=1, norm='ortho')[:, :int(conf['numcep'])]
    feat = lifter(feat, float(conf['ceplifter']))
    return feat, numpy.log(energy)


def fbank(signal, samplerate, conf):
    """
    Compute fbank features from an audio signal.

    Args:
        signal: the audio signal from which to compute features. Should be an
            N*1 array
        samplerate: the samplerate of the signal we are working with.
        conf: feature configuration

    Returns:
        A numpy array of size (NUMFRAMES by nfilt) containing features, a numpy
        vector containing the signal energy
    """

    raise BaseException('Not yet implemented')
    highfreq = int(conf['highfreq'])
    if highfreq < 0:
        highfreq = samplerate/2

    signal = sigproc.preemphasis(signal, float(conf['preemph']))
    frames = sigproc.framesig(signal, float(conf['winlen'])*samplerate,
                              float(conf['winstep'])*samplerate)
    pspec = sigproc.powspec(frames, int(conf['nfft']))

    # this stores the total energy in each frame
    energy = numpy.sum(pspec, 1)

    # if energy is zero, we get problems with log
    energy = numpy.where(energy == 0, numpy.finfo(float).eps, energy)

    filterbank = get_filterbanks(int(conf['nfilt']), int(conf['nfft']),
                                 samplerate, int(conf['lowfreq']), highfreq)

    # compute the filterbank energies
    feat = numpy.dot(pspec, filterbank.T)

    # if feat is zero, we get problems with log
    feat = numpy.where(feat == 0, numpy.finfo(float).eps, feat)

    return feat, energy


def logfbank(signal, samplerate, conf):
    """
    Compute log-fbank features from an audio signal.

    Args:
        signal: the audio signal from which to compute features. Should be an
            N*1 array
        samplerate: the samplerate of the signal we are working with.
        conf: feature configuration

    Returns:
        A numpy array of size (NUMFRAMES by nfilt) containing features, a numpy
        vector containing the signal log-energy
    """
    raise BaseException('Not yet implemented')
    feat, energy = fbank(signal, samplerate, conf)
    return numpy.log(feat), numpy.log(energy)


def ssc(signal, samplerate, conf):
    """
    Compute ssc features from an audio signal.

    Args:
        signal: the audio signal from which to compute features. Should be an
            N*1 array
        samplerate: the samplerate of the signal we are working with.
        conf: feature configuration

    Returns:
        A numpy array of size (NUMFRAMES by nfilt) containing features, a numpy
        vector containing the signal log-energy
    """
    raise BaseException('Not yet implemented')

    highfreq = int(conf['highfreq'])
    if highfreq < 0:
        highfreq = samplerate/2
    signal = sigproc.preemphasis(signal, float(conf['preemph']))
    frames = sigproc.framesig(signal, float(conf['winlen'])*samplerate,
                              float(conf['winstep'])*samplerate)
    pspec = sigproc.powspec(frames, int(conf['nfft']))

    # this stores the total energy in each frame
    energy = numpy.sum(pspec, 1)

    # if energy is zero, we get problems with log
    energy = numpy.where(energy == 0, numpy.finfo(float).eps, energy)

    filterbank = get_filterbanks(int(conf['nfilt']), int(conf['nfft']),
                                 samplerate, int(conf['lowfreq']), highfreq)

    # compute the filterbank energies
    feat = numpy.dot(pspec, filterbank.T)
    tiles = numpy.tile(numpy.linspace(1, samplerate/2, numpy.size(pspec, 1)),
                       (numpy.size(pspec, 0), 1))

    return numpy.dot(pspec*tiles, filterbank.T) / feat, numpy.log(energy)


def hz2mel(rate):
    """
    Convert a value in Hertz to Mels

    Args:
        rate: a value in Hz. This can also be a numpy array, conversion proceeds
            element-wise.

    Returns:
        a value in Mels. If an array was passed in, an identical sized array is
        returned.
    """
    return 2595 * numpy.log10(1+rate/700.0)


def mel2hz(mel):
    """
    Convert a value in Mels to Hertz

    Args:
        mel: a value in Mels. This can also be a numpy array, conversion
            proceeds element-wise.

    Returns:
        a value in Hertz. If an array was passed in, an identical sized array is
        returned.
    """
    return 700*(10**(mel/2595.0)-1)


def get_filterbanks(nfilt=20, nfft=512, samplerate=16000, lowfreq=0,
                    highfreq=None):
    """
    Compute a Mel-filterbank.

    The filters are stored in the rows, the columns correspond to fft bins.
    The filters are returned as an array of size nfilt * (nfft/2 + 1)

    Args:
        nfilt: the number of filters in the filterbank, default 20.
        nfft: the FFT size. Default is 512.
        samplerate: the samplerate of the signal we are working with. Affects
            mel spacing.
        lowfreq: lowest band edge of mel filters, default 0 Hz
        highfreq: highest band edge of mel filters, default samplerate/2

    Returns:
        A numpy array of size nfilt * (nfft/2 + 1) containing filterbank. Each
        row holds 1 filter.
    """

    raise BaseException('Not yet implemented')
    highfreq = highfreq or samplerate/2
    assert highfreq <= samplerate/2, "highfreq is greater than samplerate/2"

    # compute points evenly spaced in mels
    lowmel = hz2mel(lowfreq)
    highmel = hz2mel(highfreq)
    melpoints = numpy.linspace(lowmel, highmel, nfilt+2)

    # our points are in Hz, but we use fft bins, so we have to convert
    #  from Hz to fft bin number
    bins = numpy.floor((nfft+1)*mel2hz(melpoints)/samplerate)

    fbanks = numpy.zeros([nfilt, nfft/2+1])
    for j in xrange(0, nfilt):
        for i in xrange(int(bins[j]), int(bins[j+1])):
            fbanks[j, i] = (i - bins[j])/(bins[j+1]-bins[j])
        for i in xrange(int(bins[j+1]), int(bins[j+2])):
            fbanks[j, i] = (bins[j+2]-i)/(bins[j+2]-bins[j+1])
    return fbanks


def lifter(cepstra, liftering=22.0):
    """
    Apply a cepstral lifter the the matrix of cepstra.

    This has the effect of increasing the magnitude of the high frequency DCT
    coeffs.

    Args:
        cepstra: the matrix of mel-cepstra, will be numframes * numcep in size.
        liftering: the liftering coefficient to use. Default is 22. L <= 0
            disables lifter.

    Returns:
        the lifted cepstra
    """
    raise BaseException('Not yet implemented')
    if liftering > 0:
        _, ncoeff = numpy.shape(cepstra)
        lift = 1+(liftering/2)*numpy.sin(numpy.pi * numpy.arange(ncoeff)/liftering)
        return lift*cepstra
    else:
        # values of liftering <= 0, do nothing
        return cepstra


def deriv(features):
    """
    Compute the first order derivative of the features

    Args:
        features: the input features

    Returns:
        the firs order derivative
    """
    return convolve1d(features, numpy.array([2, 1, 0, -1, -2]), 0)


def delta(features):
    """
    concatenate the first order derivative to the features

    Args:
        features: the input features

    Returns:
        the features concatenated with the first order derivative
    """
    return numpy.concatenate((features, deriv(features)), 1)


def ddelta(features):
    """
    concatenate the first and second order derivative to the features

    Args:
        features: the input features

    Returns:
        the features concatenated with the first and second order derivative
    """
    deltafeat = deriv(features)
    return numpy.concatenate((features, deltafeat, deriv(deltafeat)), 1)
