"""Pure-numpy math primitives shared by the indicator functions.

These have no pandas dependency and can be reused outside the
indicators submodule.  All conventions per the spec at
``docs/metacoupling_codex_prompts_and_formulas_with_llm.md``
(Sections 7-8) and the established literature:

- Shannon (1948) entropy, normalised to [0, 1] by dividing by ln(n)
- Hirschman (1945) / Herfindahl (1950) HHI, normalised per
  Cracau & Lima (2016)
"""

from __future__ import annotations

import math
import warnings

import numpy as np


def shannon_entropy_normalised(shares: np.ndarray) -> float:
    """Normalised Shannon entropy of a non-negative shares array.

    Returns a value in ``[0, 1]``:
    - ``1`` when shares are perfectly even (all equal to ``1/n``)
    - ``0`` when one share carries everything

    Convention: ``0 * ln(0) = 0`` (the standard entropy convention,
    implemented via ``np.where`` so we never call ``ln(0)``).

    Parameters
    ----------
    shares:
        1-D array of non-negative shares.  Expected to sum to ~1
        but not required; the formula uses raw share values.

    Returns
    -------
    float
        Normalised entropy in ``[0, 1]``, or ``nan`` when:
        - ``shares`` is empty
        - all shares are zero (no information to measure)
        - ``len(shares) < 2`` (entropy of a 1-element distribution
          is trivially 0 / 0)
    """
    arr = np.asarray(shares, dtype=float)
    if arr.size == 0:
        return float("nan")
    if arr.size < 2:
        # ln(1) = 0 in the denominator -> undefined.  A single
        # category has no spread to measure.
        return float("nan")
    total = arr.sum()
    if total <= 0:
        return float("nan")
    # Per-element p ln(p) with the 0 ln 0 = 0 convention.  Mask zeros
    # to a benign 1 (ln(1) = 0) before taking the log to avoid NumPy's
    # divide-by-zero / uninitialised-memory warnings; then re-zero the
    # contributions of the original zeros.
    mask = arr > 0
    safe_log = np.zeros_like(arr)
    safe_log[mask] = arr[mask] * np.log(arr[mask])
    return float(-safe_log.sum() / math.log(arr.size))


def normalised_hhi(shares: np.ndarray) -> float:
    """Normalised Herfindahl-Hirschman Index of a shares array.

    Returns a value in ``[0, 1]``:
    - ``0`` when shares are perfectly even
    - ``1`` when one share carries everything

    Per Cracau & Lima (2016): ``HHI_norm = (HHI - 1/n) / (1 - 1/n)``
    where ``HHI = sum(p_i^2)``.

    Single-partner convention (per spec Section 8 edge cases):
    when ``len(shares) == 1`` the normalisation denominator is zero,
    so we define ``HHI_norm = 1`` (all flow concentrated in one
    partner) and emit a ``UserWarning``.

    Empty / all-zero shares return ``nan`` (no flow to measure).
    """
    arr = np.asarray(shares, dtype=float)
    if arr.size == 0:
        return float("nan")
    if arr.size == 1:
        warnings.warn(
            "normalised_hhi: only one partner present; concentration is "
            "defined as 1 by convention.",
            UserWarning,
            stacklevel=2,
        )
        return 1.0
    total = arr.sum()
    if total <= 0:
        return float("nan")
    p = arr / total
    hhi = float((p * p).sum())
    n = arr.size
    return (hhi - 1.0 / n) / (1.0 - 1.0 / n)


def raw_hhi(shares: np.ndarray) -> float:
    """Raw (un-normalised) Herfindahl-Hirschman Index.

    Companion to ``normalised_hhi`` -- ``ENP = 1 / HHI_raw`` gives
    the effective number of equally-sized partners (spec §9.2).
    Empty / all-zero shares return ``nan``.
    """
    arr = np.asarray(shares, dtype=float)
    if arr.size == 0:
        return float("nan")
    total = arr.sum()
    if total <= 0:
        return float("nan")
    p = arr / total
    return float((p * p).sum())
