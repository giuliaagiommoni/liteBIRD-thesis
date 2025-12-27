# mylib/utils.py
"""
Functions to add 1/f noise to the TOD.

Contains:
- add_lbsNoise: Uses litebird_sim.add_noise in blocks
- add_OofaNoise: Uses ducc0.misc.OofaNoise to generate 1/f noise for each detector
"""

import numpy as np
import litebird_sim as lbs
import ducc0


def add_lbsNoise(tod, det, block_duration_s):
    """
    Add 1/f noise using litebird_sim.add_noise in blocks.

    Parameters
    ----------
    tod : numpy.ndarray
        Array 2D shape (D, N) (D detectors, N samples). Edited in-place.
    det : object
        Object with attributes: sampling_rate_hz, fknee_mhz, fmin_hz, alpha, net_ukrts.
    block_duration_s : float
        Block size in seconds.

    Returns
    -------
    numpy.ndarray
        The modified tod (same reference).
    """
    n_block = int(block_duration_s * det.sampling_rate_hz)
    n_samp = tod.shape[1]

    dets_random = [np.random.default_rng() for _ in range(tod.shape[0])]

    for start in range(0, n_samp, n_block):
        end = min(start + n_block, n_samp)

        lbs.add_noise(
            tod=tod[:, start:end],
            noise_type="one_over_f",
            sampling_rate_hz=det.sampling_rate_hz,
            net_ukrts=det.net_ukrts,
            fknee_mhz=det.fknee_mhz,
            fmin_hz=det.fmin_hz,
            alpha=det.alpha,
            dets_random=dets_random,
            scale=1.0,
        )

    return tod


def add_OofaNoise(tod, det, block_duration_s):
    """
    Add 1/f (Oofa) noise per-detector using ducc0.misc.OofaNoise.

    Parameters
    ----------
    tod : numpy.ndarray
        Array 2D shape (D, N). Edited in-place.
    det : object
        Object with attributes: sampling_rate_hz, fknee_mhz, fmin_hz, alpha, net_ukrts.
    block_duration_s : float
        Block size in seconds.

    Returns
    -------
    numpy.ndarray
        The modified tod (same reference).
    """
    n_block = int(block_duration_s * det.sampling_rate_hz)
    n_samp = tod.shape[1]

    noise_gens = [
        ducc0.misc.OofaNoise(
            sigmawhite=det.net_ukrts * 1e-6 * np.sqrt(det.sampling_rate_hz),
            f_knee=det.fknee_mhz * 1e-3,
            f_min=det.fmin_hz,
            f_samp=det.sampling_rate_hz,
            slope=-det.alpha,
        )
        for _ in range(tod.shape[0])
    ]

    for start in range(0, n_samp, n_block):
        end = min(start + n_block, n_samp)
        block_len = end - start

        for d in range(tod.shape[0]):
            white_chunk = np.random.normal(0.0, 1.0, block_len)
            tod[d, start:end] += noise_gens[d].filterGaussian(white_chunk)
            del white_chunk

    return tod
