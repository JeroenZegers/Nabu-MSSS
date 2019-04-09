#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Source separation algorithms attempt to extract recordings of individual
sources from a recording of a mixture of sources.  Evaluation methods for
source separation compare the extracted sources from reference sources and
attempt to measure the perceptual quality of the separation.

See also the bss_eval MATLAB toolbox:
    http://bass-db.gforge.inria.fr/bss_eval/

Conventions
-----------

An audio signal is expected to be in the format of a 1-dimensional array where
the entries are the samples of the audio signal.  When providing a group of
estimated or reference sources, they should be provided in a 2-dimensional
array, where the first dimension corresponds to the source number and the
second corresponds to the samples.

Metrics
-------

* :func:`mir_eval.separation.bss_eval_sources`: Computes the bss_eval_sources
  metrics from bss_eval, which optionally optimally match the estimated sources
  to the reference sources and measure the distortion and artifacts present in
  the estimated sources as well as the interference between them.

* :func:`mir_eval.separation.bss_eval_sources_framewise`: Computes the
  bss_eval_sources metrics on a frame-by-frame basis.

* :func:`mir_eval.separation.bss_eval_images`: Computes the bss_eval_images
  metrics from bss_eval, which includes the metrics in
  :func:`mir_eval.separation.bss_eval_sources` plus the image to spatial
  distortion ratio.

* :func:`mir_eval.separation.bss_eval_images_framewise`: Computes the
  bss_eval_images metrics on a frame-by-frame basis.

References
----------
  .. [#vincent2006performance] Emmanuel Vincent, Rémi Gribonval, and Cédric
      Févotte, "Performance measurement in blind audio source separation," IEEE
      Trans. on Audio, Speech and Language Processing, 14(4):1462-1469, 2006.


This code is licensed under the MIT License:
The MIT License (MIT)

Copyright (c) 2014 Colin Raffel

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

Please see http://craffel.github.io/mir_eval/ for more information

Addition by Pieter Appeltans on 4/4/2018: bss_eval for noisy signals
'''

import numpy as np
import scipy.fftpack
from scipy.linalg import toeplitz
from scipy.signal import fftconvolve
import collections
import itertools
import warnings
#from . import util


# The maximum allowable number of sources (prevents insane computational load)
MAX_SOURCES = 100


def filter_kwargs(_function, *args, **kwargs):
    """Given a function and args and keyword args to pass to it, call the function
    but using only the keyword arguments which it accepts.  This is equivalent
    to redefining the function with an additional \*\*kwargs to accept slop
    keyword args.

    If the target function already accepts \*\*kwargs parameters, no filtering
    is performed.

    Parameters
    ----------
    _function : callable
        Function to call.  Can take in any number of args or kwargs

    """

    if has_kwargs(_function):
        return _function(*args, **kwargs)

    # Get the list of function arguments
    func_code = six.get_function_code(_function)
    function_args = func_code.co_varnames[:func_code.co_argcount]
    # Construct a dict of those kwargs which appear in the function
    filtered_kwargs = {}
    for kwarg, value in list(kwargs.items()):
        if kwarg in function_args:
            filtered_kwargs[kwarg] = value
    # Call the function with the supplied args and the filtered kwarg dict
    return _function(*args, **filtered_kwargs)

def validate(reference_sources, estimated_sources):
    """Checks that the input data to a metric are valid, and throws helpful
    errors if not.

    Parameters
    ----------
    reference_sources : np.ndarray, shape=(nsrc, nsampl)
        matrix containing true sources
    estimated_sources : np.ndarray, shape=(nsrc, nsampl)
        matrix containing estimated sources

    """

    if reference_sources.shape != estimated_sources.shape:
        raise ValueError('The shape of estimated sources and the true '
                         'sources should match.  reference_sources.shape '
                         '= {}, estimated_sources.shape '
                         '= {}'.format(reference_sources.shape,
                                       estimated_sources.shape))

    if reference_sources.ndim > 3 or estimated_sources.ndim > 3:
        raise ValueError('The number of dimensions is too high (must be less '
                         'than 3). reference_sources.ndim = {}, '
                         'estimated_sources.ndim '
                         '= {}'.format(reference_sources.ndim,
                                       estimated_sources.ndim))

    if reference_sources.size == 0:
        warnings.warn("reference_sources is empty, should be of size "
                      "(nsrc, nsample).  sdr, sir, sar, and perm will all "
                      "be empty np.ndarrays")
    elif _any_source_silent(reference_sources):
        raise ValueError('All the reference sources should be non-silent (not '
                         'all-zeros), but at least one of the reference '
                         'sources is all 0s, which introduces ambiguity to the'
                         ' evaluation. (Otherwise we can add infinitely many '
                         'all-zero sources.)')

    if estimated_sources.size == 0:
        warnings.warn("estimated_sources is empty, should be of size "
                      "(nsrc, nsample).  sdr, sir, sar, and perm will all "
                      "be empty np.ndarrays")
    elif _any_source_silent(estimated_sources):
        raise ValueError('All the estimated sources should be non-silent (not '
                         'all-zeros), but at least one of the estimated '
                         'sources is all 0s. Since we require each reference '
                         'source to be non-silent, having a silent estimated '
                         'source will result in an underdetermined system.')

    if (estimated_sources.shape[0] > MAX_SOURCES or
            reference_sources.shape[0] > MAX_SOURCES):
        raise ValueError('The supplied matrices should be of shape (nsrc,'
                         ' nsampl) but reference_sources.shape[0] = {} and '
                         'estimated_sources.shape[0] = {} which is greater '
                         'than mir_eval.separation.MAX_SOURCES = {}.  To '
                         'override this check, set '
                         'mir_eval.separation.MAX_SOURCES to a '
                         'larger value.'.format(reference_sources.shape[0],
                                                estimated_sources.shape[0],
                                                MAX_SOURCES))
def validate_extended(reference_sources, estimated_sources,noise_source):
    """Checks that the input data to a metric are valid, and throws helpful
    errors if not.

    Parameters
    ----------
    reference_sources : np.ndarray, shape=(nsrc, nsampl)
        matrix containing true sources
    estimated_sources : np.ndarray, shape=(nsrc, nsampl)
        matrix containing estimated sources

    """

    validate(reference_sources, estimated_sources)
    if noise_source.size == 0:
        warnings.warn("noise_source is empty, should be of size "
                      "(1, nsample).  sdr, sir, sar, snr, and perm will all "
                      "be empty np.ndarrays")
def _project_extended(reference_sources, estimated_source,noise_source, flen):
    """Least-squares projection of estimated source on the subspace spanned by
    delayed versions of reference sources, with delays between 0 and flen-1
    """
    print np.shape(noise_source),np.shape(estimated_source)
    nsrc = reference_sources.shape[0]
    nsampl = reference_sources.shape[1]

    # computing coefficients of least squares problem via FFT ##
    # zero padding and FFT of input data
    reference_sources = np.hstack((reference_sources,
                                   np.zeros((nsrc, flen - 1))))
    
    noise_source = np.hstack((noise_source,np.zeros(flen-1)))
    sources = np.vstack((reference_sources,noise_source))
    estimated_source = np.hstack((estimated_source, np.zeros(flen - 1)))
    n_fft = int(2**np.ceil(np.log2(nsampl + flen - 1.)))
    sf = scipy.fftpack.fft(sources, n=n_fft, axis=1)
    sef = scipy.fftpack.fft(estimated_source, n=n_fft)
    # inner products between delayed versions of reference_sources
    G = np.zeros(((nsrc+1) * flen, (nsrc+1) * flen))
    for i in range(nsrc+1):
        for j in range(nsrc+1):
            ssf = sf[i] * np.conj(sf[j])
            ssf = np.real(scipy.fftpack.ifft(ssf))
            ss = toeplitz(np.hstack((ssf[0], ssf[-1:-flen:-1])),
                          r=ssf[:flen])
            G[i * flen: (i+1) * flen, j * flen: (j+1) * flen] = ss
            G[j * flen: (j+1) * flen, i * flen: (i+1) * flen] = ss.T
    # inner products between estimated_source and delayed versions of
    # reference_sources
    D = np.zeros((nsrc+1) * flen)
    for i in range(nsrc+1):
        ssef = sf[i] * np.conj(sef)
        ssef = np.real(scipy.fftpack.ifft(ssef))
        D[i * flen: (i+1) * flen] = np.hstack((ssef[0], ssef[-1:-flen:-1]))

    # Computing projection
    # Distortion filters
    try:
        C = np.linalg.solve(G, D).reshape(flen, nsrc, order='F')
    except np.linalg.linalg.LinAlgError:
        C = np.linalg.lstsq(G, D)[0].reshape(flen, nsrc, order='F')
    # Filtering
    sproj = np.zeros(nsampl + flen - 1)
    for i in range(nsrc+1):
        sproj += fftconvolve(C[:, i], sources[i])[:nsampl + flen - 1]
    return sproj

def _project(reference_sources, estimated_source, flen):
    """Least-squares projection of estimated source on the subspace spanned by
    delayed versions of reference sources, with delays between 0 and flen-1
    """
    nsrc = reference_sources.shape[0]
    nsampl = reference_sources.shape[1]

    # computing coefficients of least squares problem via FFT ##
    # zero padding and FFT of input data
    reference_sources = np.hstack((reference_sources,
                                   np.zeros((nsrc, flen - 1))))
    estimated_source = np.hstack((estimated_source, np.zeros(flen - 1)))
    n_fft = int(2**np.ceil(np.log2(nsampl + flen - 1.)))
    sf = scipy.fftpack.fft(reference_sources, n=n_fft, axis=1)
    sef = scipy.fftpack.fft(estimated_source, n=n_fft)
    # inner products between delayed versions of reference_sources
    G = np.zeros((nsrc * flen, nsrc * flen))
    for i in range(nsrc):
        for j in range(nsrc):
            ssf = sf[i] * np.conj(sf[j])
            ssf = np.real(scipy.fftpack.ifft(ssf))
            ss = toeplitz(np.hstack((ssf[0], ssf[-1:-flen:-1])),
                          r=ssf[:flen])
            G[i * flen: (i+1) * flen, j * flen: (j+1) * flen] = ss
            G[j * flen: (j+1) * flen, i * flen: (i+1) * flen] = ss.T
    # inner products between estimated_source and delayed versions of
    # reference_sources
    D = np.zeros(nsrc * flen)
    for i in range(nsrc):
        ssef = sf[i] * np.conj(sef)
        ssef = np.real(scipy.fftpack.ifft(ssef))
        D[i * flen: (i+1) * flen] = np.hstack((ssef[0], ssef[-1:-flen:-1]))

    # Computing projection
    # Distortion filters
    try:
        C = np.linalg.solve(G, D).reshape(flen, nsrc, order='F')
    except np.linalg.linalg.LinAlgError:
        C = np.linalg.lstsq(G, D)[0].reshape(flen, nsrc, order='F')
    # Filtering
    sproj = np.zeros(nsampl + flen - 1)
    for i in range(nsrc):
        sproj += fftconvolve(C[:, i], reference_sources[i])[:nsampl + flen - 1]
    return sproj


def _project_images(reference_sources, estimated_source, flen, G=None):
    """Least-squares projection of estimated source on the subspace spanned by
    delayed versions of reference sources, with delays between 0 and flen-1.
    Passing G as all zeros will populate the G matrix and return it so it can
    be passed into the next call to avoid recomputing G (this will only works
    if not computing permutations).
    """
    nsrc = reference_sources.shape[0]
    nsampl = reference_sources.shape[1]
    nchan = reference_sources.shape[2]
    reference_sources = np.reshape(np.transpose(reference_sources, (2, 0, 1)),
                                   (nchan*nsrc, nsampl), order='F')

    # computing coefficients of least squares problem via FFT ##
    # zero padding and FFT of input data
    reference_sources = np.hstack((reference_sources,
                                   np.zeros((nchan*nsrc, flen - 1))))
    estimated_source = \
        np.hstack((estimated_source.transpose(), np.zeros((nchan, flen - 1))))
    n_fft = int(2**np.ceil(np.log2(nsampl + flen - 1.)))
    sf = scipy.fftpack.fft(reference_sources, n=n_fft, axis=1)
    sef = scipy.fftpack.fft(estimated_source, n=n_fft)

    # inner products between delayed versions of reference_sources
    if G is None:
        saveg = False
        G = np.zeros((nchan * nsrc * flen, nchan * nsrc * flen))
        for i in range(nchan * nsrc):
            for j in range(i+1):
                ssf = sf[i] * np.conj(sf[j])
                ssf = np.real(scipy.fftpack.ifft(ssf))
                ss = toeplitz(np.hstack((ssf[0], ssf[-1:-flen:-1])),
                              r=ssf[:flen])
                G[i * flen: (i+1) * flen, j * flen: (j+1) * flen] = ss
                G[j * flen: (j+1) * flen, i * flen: (i+1) * flen] = ss.T
    else:  # avoid recomputing G (only works if no permutation is desired)
        saveg = True  # return G
        if np.all(G == 0):  # only compute G if passed as 0
            G = np.zeros((nchan * nsrc * flen, nchan * nsrc * flen))
            for i in range(nchan * nsrc):
                for j in range(i+1):
                    ssf = sf[i] * np.conj(sf[j])
                    ssf = np.real(scipy.fftpack.ifft(ssf))
                    ss = toeplitz(np.hstack((ssf[0], ssf[-1:-flen:-1])),
                                  r=ssf[:flen])
                    G[i * flen: (i+1) * flen, j * flen: (j+1) * flen] = ss
                    G[j * flen: (j+1) * flen, i * flen: (i+1) * flen] = ss.T

    # inner products between estimated_source and delayed versions of
    # reference_sources
    D = np.zeros((nchan * nsrc * flen, nchan))
    for k in range(nchan * nsrc):
        for i in range(nchan):
            ssef = sf[k] * np.conj(sef[i])
            ssef = np.real(scipy.fftpack.ifft(ssef))
            D[k * flen: (k+1) * flen, i] = \
                np.hstack((ssef[0], ssef[-1:-flen:-1])).transpose()

    # Computing projection
    # Distortion filters
    try:
        C = np.linalg.solve(G, D).reshape(flen, nchan*nsrc, nchan, order='F')
    except np.linalg.linalg.LinAlgError:
        C = np.linalg.lstsq(G, D)[0].reshape(flen, nchan*nsrc, nchan,
                                             order='F')
    # Filtering
    sproj = np.zeros((nchan, nsampl + flen - 1))
    for k in range(nchan * nsrc):
        for i in range(nchan):
            sproj[i] += fftconvolve(C[:, k, i].transpose(),
                                    reference_sources[k])[:nsampl + flen - 1]
    # return G only if it was passed in
    if saveg:
        return sproj, G
    else:
        return sproj

def _bss_source_crit_extended(s_true, e_spat, e_interf, e_noise, e_artif):
    """Measurement of the separation quality for a given source in terms of
    filtered true source, interference and artifacts.
    """
    # energy ratios
    s_filt = s_true + e_spat
    sdr = _safe_db(np.sum(s_filt**2), np.sum((e_interf + e_noise + e_artif)**2))
    sir = _safe_db(np.sum(s_filt**2), np.sum(e_interf**2))
    snr = _safe_db(np.sum((s_filt+e_interf)**2), np.sum(e_noise**2))
    sar = _safe_db(np.sum((s_filt + e_interf + e_noise)**2), np.sum(e_artif**2))
    return (sdr, sir,snr, sar)

def _bss_source_crit(s_true, e_spat, e_interf, e_artif):
    """Measurement of the separation quality for a given source in terms of
    filtered true source, interference and artifacts.
    """
    # energy ratios
    s_filt = s_true + e_spat
    sdr = _safe_db(np.sum(s_filt**2), np.sum((e_interf + e_artif)**2))
    sir = _safe_db(np.sum(s_filt**2), np.sum(e_interf**2))
    sar = _safe_db(np.sum((s_filt + e_interf)**2), np.sum(e_artif**2))
    return (sdr, sir, sar)


def _bss_image_crit(s_true, e_spat, e_interf, e_artif):
    """Measurement of the separation quality for a given image in terms of
    filtered true source, spatial error, interference and artifacts.
    """
    # energy ratios
    sdr = _safe_db(np.sum(s_true**2), np.sum((e_spat+e_interf+e_artif)**2))
    isr = _safe_db(np.sum(s_true**2), np.sum(e_spat**2))
    sir = _safe_db(np.sum((s_true+e_spat)**2), np.sum(e_interf**2))
    sar = _safe_db(np.sum((s_true+e_spat+e_interf)**2), np.sum(e_artif**2))
    return (sdr, isr, sir, sar)


def _safe_db(num, den):
    """Properly handle the potential +Inf db SIR, instead of raising a
    RuntimeWarning. Only denominator is checked because the numerator can never
    be 0.
    """
    if den == 0:
        return np.Inf
    return 10 * np.log10(num / den)


def evaluate(reference_sources, estimated_sources, **kwargs):
    """Compute all metrics for the given reference and estimated signals.

    NOTE: This will always compute :func:`mir_eval.separation.bss_eval_images`
    for any valid input and will additionally compute
    :func:`mir_eval.separation.bss_eval_sources` for valid input with fewer
    than 3 dimensions.

    Examples
    --------
    >>> # reference_sources[n] should be an ndarray of samples of the
    >>> # n'th reference source
    >>> # estimated_sources[n] should be the same for the n'th estimated source
    >>> scores = mir_eval.separation.evaluate(reference_sources,
    ...                                       estimated_sources)

    Parameters
    ----------
    reference_sources : np.ndarray, shape=(nsrc, nsampl[, nchan])
        matrix containing true sources
    estimated_sources : np.ndarray, shape=(nsrc, nsampl[, nchan])
        matrix containing estimated sources
    kwargs
        Additional keyword arguments which will be passed to the
        appropriate metric or preprocessing functions.

    Returns
    -------
    scores : dict
        Dictionary of scores, where the key is the metric name (str) and
        the value is the (float) score achieved.

    """
    # Compute all the metrics
    scores = collections.OrderedDict()

    sdr, isr, sir, sar, perm = filter_kwargs(
        bss_eval_images,
        reference_sources,
        estimated_sources,
        **kwargs
    )
    scores['Images - Source to Distortion'] = sdr.tolist()
    scores['Images - Image to Spatial'] = isr.tolist()
    scores['Images - Source to Interference'] = sir.tolist()
    scores['Images - Source to Artifact'] = sar.tolist()
    scores['Images - Source permutation'] = perm.tolist()

    sdr, isr, sir, sar, perm = filter_kwargs(
        bss_eval_images_framewise,
        reference_sources,
        estimated_sources,
        **kwargs
    )
    scores['Images Frames - Source to Distortion'] = sdr.tolist()
    scores['Images Frames - Image to Spatial'] = isr.tolist()
    scores['Images Frames - Source to Interference'] = sir.tolist()
    scores['Images Frames - Source to Artifact'] = sar.tolist()
    scores['Images Frames - Source permutation'] = perm.tolist()

    # Verify we can compute sources on this input
    if reference_sources.ndim < 3 and estimated_sources.ndim < 3:
        sdr, sir, sar, perm = filter_kwargs(
            bss_eval_sources_framewise,
            reference_sources,
            estimated_sources,
            **kwargs
        )
        scores['Sources Frames - Source to Distortion'] = sdr.tolist()
        scores['Sources Frames - Source to Interference'] = sir.tolist()
        scores['Sources Frames - Source to Artifact'] = sar.tolist()
        scores['Sources Frames - Source permutation'] = perm.tolist()

        sdr, sir, sar, perm = filter_kwargs(
            bss_eval_sources,
            reference_sources,
            estimated_sources,
            **kwargs
        )
        scores['Sources - Source to Distortion'] = sdr.tolist()
        scores['Sources - Source to Interference'] = sir.tolist()
        scores['Sources - Source to Artifact'] = sar.tolist()
        scores['Sources - Source permutation'] = perm.tolist()

    return scores

if __name__=="__main__":
    # Simple demo
    ts = np.linspace(0,5,10000)
    srcs = np.array([np.sin(ts*600),
                     np.cos(357*ts+0.01)])
    recons = srcs[::-1] + np.random.randn(*srcs.shape)*2

    sdr, sir, sar, permut = bss_eval_sources(srcs, recons)
    print("""SDR: {}
SIR: {}
SAR: {}""".format(sdr, sir, sar))
