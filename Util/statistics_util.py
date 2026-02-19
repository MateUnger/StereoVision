import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import pandas as pd
import pingouin as pg


def remove_nan_positions(arr1: np.ndarray, arr2: np.ndarray) -> tuple:
    """
    Remove Nan elements from both input arrays (of equal length).
    Elements are only kepth if at position P both arrays have valid values.

    Args:
        arr1: first input array
        arr2: second input array

    Returns:
        arr1_cleaned: arr1 containing elements where both arr1 and arr2 are valid (not Nan)
        arr2_cleaned: arr2 ....same as above

    """
    if arr1.shape != arr2.shape:
        raise Exception(
            f"Input arrays must have the same shape. array_1: {arr1.shape}, array2: {arr2.shape}"
        )

    # Find positions of NaNs in both arrays
    nan_positions_arr1 = np.isnan(arr1)
    nan_positions_arr2 = np.isnan(arr2)

    # Combine positions to find indices to remove
    nan_positions_combined = nan_positions_arr1 | nan_positions_arr2

    # Filter out the NaN positions from both arrays
    arr1_cleaned = arr1[~nan_positions_combined]
    arr2_cleaned = arr2[~nan_positions_combined]

    return arr1_cleaned, arr2_cleaned


def bland_altman_statistics(method_a: np.ndarray, method_b: np.ndarray, plot=False) -> tuple:
    """
    Calculate Bland-Altman statistics and optionally plot the Bland-Altman plot.

    Args:
        method_a: Array-like, measurements from method A.
        method_b: Array-like, measurements from method B.
        plot: Boolean, if True, plots the Bland-Altman plot.

    Returns:
        bias: Mean difference between the methods.
        rpc: Reproducibility coefficient (1.96 * standard deviation of differences).
        cv: Coefficient of variation.
    """

    # Calculating differences and means
    differences = method_b - method_a
    mean_difference = np.mean(differences)
    std_dev_difference = np.std(differences, ddof=1)
    means = (method_a + method_b) / 2

    # Bias (Mean Difference)
    bias = mean_difference

    # Reproducibility Coefficient (RPC)
    rpc = 1.96 * std_dev_difference

    # Coefficient of Variation (CV)
    cv = (std_dev_difference / abs(bias)) * 100

    # Bland-Altman Plot
    if plot:
        plt.scatter(means, differences)
        plt.axhline(mean_difference, color="gray", linestyle="--")
        plt.axhline(mean_difference + rpc, color="gray", linestyle="--")
        plt.axhline(mean_difference - rpc, color="gray", linestyle="--")
        plt.title("Bland-Altman Plot")
        plt.xlabel("Mean of Two Methods")
        plt.ylabel("Difference Between Methods")
        plt.show()

        # Printing the calculated values
        print(f"Bias (Mean Difference): {bias}")
        print(f"Reproducibility Coefficient (RPC): {rpc}")
        print(f"Coefficient of Variation (CV): {cv:.2f}%")

    return bias, rpc, cv


def uniform_statistics(ground_truth_measurements, new_system_measurements):
    """
    Calculate Pearson correlation coefficient, Bland-Altman statistics, and ICC between two methods.

    Parameters:
    - ground_truth_measurement: Array-like, measurements from method A (ground truth).
    - new_system_measurements:  Array-like, measurements from method B (new measurement system).

    Returns:
    - correlation_coefficient: Pearson correlation coefficient between the two methods.
    - p_value: P-value for the Pearson correlation.
    - bias: Mean difference between the methods.
    - rpc: Reproducibility coefficient.
    - cv: Coefficient of variation.
    - icc_results: DataFrame with ICC results.
    """

    ground_truth_data = np.array(ground_truth_measurements)
    new_system_data = np.array(new_system_measurements)
    # print(ground_truth_data.shape, new_system_data.shape)

    # Filter arrays for NaNs
    ground_truth_data, new_system_data = remove_nan_positions(ground_truth_data, new_system_data)

    # absolute error
    absolute_error = np.nanmean(np.abs(ground_truth_data - new_system_data))
    # relative error
    relative_error = (
        np.nanmean(np.abs((ground_truth_data - new_system_data) / ground_truth_data)) * 100
    )

    # Calculate RMSE
    rmse = np.nanmean(np.sqrt(np.nanmean((ground_truth_data - new_system_data) ** 2)))
    # relative RMSE
    relative_rmse = rmse / np.nanmean(ground_truth_data)

    mean_gt = np.nanmean(ground_truth_data)
    mean_ns = np.nanmean(new_system_data)
    std_gt = np.nanstd(ground_truth_data)
    std_ns = np.nanstd(new_system_data)

    try:
        # Calculate Pearson correlation coefficient and p-value
        correlation_coefficient, p_value = pearsonr(ground_truth_data, new_system_data)
    except Exception as e:
        print(e)
    # Calculate Bland-Altman statistics
    bias, rpc, cv = bland_altman_statistics(ground_truth_data, new_system_data)

    # Calculate ICC
    icc_results = icc_statistics(ground_truth_data, new_system_data)
    # get the ICC(3,1) result from the table
    icc_3_1 = float(icc_results.loc[icc_results["Type"] == "ICC3"]["ICC"].iloc[0])

    stat_results = {
        "mean_gt": mean_gt,
        "mean_ns": mean_ns,
        "std_gt": std_gt,
        "std_ns": std_ns,
        "absolute_error": absolute_error,
        "relative_error": relative_error,
        "rmse": rmse,
        "relative_rmse": relative_rmse,
        "correlation_coefficient": correlation_coefficient,
        "p_value": p_value,
        "bias": bias,
        "rcp": rpc,
        "cv": cv,
        "icc_3_1": icc_3_1,
    }

    # # Create a formatted table
    # table = (
    #     f"Mean GT: {mean_gt:.4f}\n"
    #     f"Mean PD: {mean_pd:.4f}\n"
    #     f"Std GT: {std_gt:.4f}\n"
    #     f"Std PD: {std_pd:.4f}\n"
    #     f"Absolute Error: {absolute_error:.4f}\n"
    #     f"Relative Error: {relative_error:.4f}\n"
    #     f"RMSE: {rmse:.4f}\n"
    #     f"Relative RMSE: {relative_rmse:.4f}\n"
    #     f"Correlation Coefficient: {correlation_coefficient:.4f}\n"
    #     f"P-value: {p_value:.4f}\n"
    #     f"Bias (Mean Difference): {bias:.4f}\n"
    #     f"Reproducibility Coefficient (RPC): {rpc:.4f}\n"
    #     f"Coefficient of Variation (CV): {cv:.4f}\n"
    #     f"\nICC Results:\n{icc_results}"
    # )

    return stat_results


def icc_statistics(method_a: np.ndarray, method_b: np.ndarray) -> pd.DataFrame:
    """
    Calculate Intraclass Correlation Coefficient (ICC) between two methods.

    Args:
        method_a: Array-like, measurements from method A.
        method_b: Array-like, measurements from method B.

    Returns:

        icc_results: DataFrame with ICC results.
    """

    subjects = list(range(1, len(method_a) + 1))

    # Prepare the DataFrame
    data = pd.DataFrame(
        {
            "Subject_ID": subjects + subjects,  # Repeat subject IDs for each method
            "Measurement": list(method_a) + list(method_b),  # Combine measurements
            "Method": ["A"] * len(method_a) + ["B"] * len(method_b),  # Label methods
        }
    )

    # Convert "Method" to categorical
    data["Method"] = pd.Categorical(data["Method"])

    # Calculate ICCs
    icc_results = pg.intraclass_corr(
        data=data, targets="Subject_ID", raters="Method", ratings="Measurement"
    )

    return icc_results
