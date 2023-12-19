"""
Variable transformation for Hs, Tz, and steepness.

These transformations are used in Windmeier's EW and in the 
nonzero EW model (see doi.org/10.26092/elib/2181 and predefined.py)
"""

import numpy as np

g = 9.81
factor = 2 * np.pi / g
factor_sqrt = np.sqrt(factor)


def hs_tz_to_s_d(hs, tz):
    global factor
    s = factor * hs / (tz * tz)
    d = np.sqrt(hs * hs + tz * tz / 2)
    return s, d


def s_d_to_hs_tz(s, d):
    global factor
    global factor_sqrt
    hs = (np.sqrt(16 * d**2 * s**2 + factor**2) - factor) / (4 * s)
    tz = (
        1
        / 2
        * np.sqrt(
            (factor * np.sqrt(16 * d**2 * s**2 + factor**2)) / s**2
            - factor**2 / s**2
        )
    )
    return hs, tz


def hs_tz_to_hs_s(hs, tz):
    global factor
    s = factor * hs / (tz * tz)
    return hs, s


def hs_s_to_hs_tz(hs, s):
    global factor_sqrt
    tz = factor_sqrt * np.sqrt(hs / s)
    return hs, tz


def hs_tz_to_s_tz(hs, tz):
    global factor
    s = factor * hs / (tz * tz)
    return s, tz


def s_tz_to_hs_tz(s, tz):
    global factor
    hs = s * np.square(tz) / factor
    return hs, tz
