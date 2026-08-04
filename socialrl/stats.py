import itertools
from typing import Iterable, List

import numpy as np


def paired_sign_flip_test(a: Iterable[float], b: Iterable[float], seed: int = 0, max_permutations: int = 100_000) -> float:
    """Two-sided paired randomization test, exact for up to 16 paired seeds."""
    a_array = np.asarray(list(a), dtype=float)
    b_array = np.asarray(list(b), dtype=float)
    valid = ~(np.isnan(a_array) | np.isnan(b_array))
    differences = a_array[valid] - b_array[valid]
    if len(differences) == 0:
        return float("nan")
    observed = abs(float(np.mean(differences)))
    if len(differences) <= 16:
        signs = itertools.product((-1.0, 1.0), repeat=len(differences))
        statistics = [abs(float(np.mean(differences * np.asarray(sign)))) for sign in signs]
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice((-1.0, 1.0), size=(max_permutations, len(differences)))
        statistics = np.abs(np.mean(signs * differences[None, :], axis=1))
    return float(np.mean(np.asarray(statistics) >= observed - 1e-12))


def holm_adjust(pvalues: Iterable[float]) -> List[float]:
    """Holm-Bonferroni adjusted p-values; NaNs remain NaN."""
    values = list(pvalues)
    indexed = [(index, value) for index, value in enumerate(values) if not np.isnan(value)]
    adjusted = [float("nan")] * len(values)
    previous = 0.0
    total = len(indexed)
    for rank, (index, value) in enumerate(sorted(indexed, key=lambda pair: pair[1])):
        corrected = min(1.0, (total - rank) * value)
        corrected = max(previous, corrected)
        adjusted[index] = corrected
        previous = corrected
    return adjusted
