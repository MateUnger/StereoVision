import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, peak_widths

import copy
from typing import Literal

# -----------------------------------------STEP DETECTION---------------------------------------------


def compute_asymmetry(left_values: np.ndarray, right_values: np.ndarray) -> float:
    """
    Calculate the asymmetry between means of left and right sides for a given set of values.

    Args:
        left_values: 1D array of values of the left side
        right_values: 1D array of values of the right side

    Returns:
        Percentile difference between smaller and larger means.
        If either of the input arrays are all nans or the larger mean is not positive returns np.nan
    """
    if not (np.isnan(left_values).all() or np.isnan(right_values).all()):
        left_mean = np.nanmean(left_values)
        right_mean = np.nanmean(right_values)
        smaller = np.nanmin([left_mean, right_mean])
        larger = np.nanmax([left_mean, right_mean])
        return 100 * (1 - smaller / larger) if larger > 0 else np.nan
    return np.nan
