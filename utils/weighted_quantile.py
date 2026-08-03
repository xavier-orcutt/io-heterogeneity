import numpy as np
import pandas as pd

def weighted_quantile(values, quantiles, sample_weight=None):
    """
    Calculate weighted quantiles.

    Parameters
    ----------
    values : array-like
        Observed values.
    quantiles : array-like
        Quantiles between 0 and 1.
    sample_weight : array-like, optional
        Non-negative weights.
    """
    values = np.asarray(values)
    quantiles = np.asarray(quantiles)

    if sample_weight is None:
        sample_weight = np.ones(len(values))

    sample_weight = np.asarray(sample_weight)

    valid = (
        np.isfinite(values)
        & np.isfinite(sample_weight)
        & (sample_weight >= 0)
    )

    values = values[valid]
    sample_weight = sample_weight[valid]

    sorter = np.argsort(values)
    values = values[sorter]
    sample_weight = sample_weight[sorter]

    cumulative_weight = np.cumsum(sample_weight)
    cumulative_weight = (
        cumulative_weight - 0.5 * sample_weight
    ) / sample_weight.sum()

    return np.interp(quantiles, cumulative_weight, values)

