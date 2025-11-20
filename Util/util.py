import os
import cv2
import sys
import numpy as np
import warnings
import numpy as np
import importlib
from scipy.signal import butter, filtfilt
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

from numpy.fft import fft, ifft
from scipy import signal as sp_signal
from typing import Literal


rtmlib_module = importlib.import_module("rtmlib")


from scipy.signal import butter, filtfilt

import copy
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, peak_widths
from scipy.interpolate import CubicSpline


class Pattern:
    """
    Chessboard pattern parameters
    """

    def __init__(self, height, width, square_size):
        self.pattern_height = height
        self.pattern_width = width
        self.square_size = float(square_size)  # in world coordinate system (m,mm,...)
        self.pattern_type = "chessboard"
        self.pattern_size = (self.pattern_width, self.pattern_height)  # number of (inner) corners

        self.pattern_points = np.zeros((np.prod(self.pattern_size), 3), np.float32)
        self.pattern_points[:, :2] = np.indices(self.pattern_size).T.reshape(-1, 2)
        self.pattern_points *= square_size


def get_serial_number(filename: str) -> str:
    """
    get the camera serial number from filename

    Parameters:
        filename: should look like SN123456_...
    """

    return filename.split("_")[-1].split(".")[0]


def get_points(
    img, file_name, output_dir, pattern_params: Pattern, save_output=False
) -> np.ndarray:
    """
    Find corners of chessboard, approximate them (subpixel precision)

    Args:
        img: input image
        file_name: input filename
        output_dir: output directory for saving drawn corners
        pattern_params: parameters (height, width, ...) of the real world calibration pattern used
        save_output: save an image with the world points marked, default False

    Returns:
        image_points, object_points arrays for the input image
    """
    found = False
    corners = 0
    found, corners = cv2.findChessboardCorners(img, pattern_params.pattern_size)
    if found:
        print(f"corners found: {len(corners)}")
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 50, 0.1)
        # refine corners
        cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), term)

        # points in camera/image space
        frame_img_points = corners.reshape(-1, 2)

        # points in real world
        frame_obj_points = pattern_params.pattern_points
    else:
        print("corners not found")
        return None

    outfile = os.path.join(output_dir, file_name + "_board.png")
    if save_output:
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.drawChessboardCorners(vis, pattern_params.pattern_size, corners, found)

        # draw real world origin on image
        vis = cv2.circle(
            vis,
            (int(frame_img_points[0][0]), int(frame_img_points[0][1])),
            10,
            (255, 0, 0),
            2,
        )
        cv2.imwrite(outfile, vis)

        print(f"{outfile}... OK")

    return [frame_img_points, frame_obj_points]


def get_turns_and_perspective(
    data: np.ndarray, kpt_labels: list, sampling_fr: float, min_walk_duration: float = 2
) -> tuple:
    """
    Identifies turning segments (straight/turning) and perspective (front / back) based on shoulder coordinates using peak detection method.

    Args:
        data:               Input data with shape (n_keypoints, n_dims, n_frames).
        kpt_labels:         List of keypoint labels corresponding to data (left_ankle, nose, etc).
        fps:                sampling frequency
        min_walk_duration:  min elapsed time (in seconds) between turns (1 turn = 180 deg)

    Returns:
        turn_mask: boolean array indicating straight turning segments (1) and straight segments (0).

        perspective: bollean array indicating frontal facing (1) and back facing (0) segments.
    """
    # get the y coord of shoulders (for qualisys x-y plane is horizontal, y coord was forwards/backwards movement)
    shoulder_R = data[kpt_labels.index("right_shoulder"), 1, :]
    shoulder_L = data[kpt_labels.index("left_shoulder"), 1, :]

    shoulder_diff = np.abs(np.diff(shoulder_R - shoulder_L))
    shoulder_diff = shoulder_diff / np.nanmax(shoulder_diff)

    # Interpolate NaNs in shoulder_diff
    if np.any(np.isnan(shoulder_diff)):
        nans = np.isnan(shoulder_diff)
        not_nans = ~nans
        shoulder_diff[nans] = np.interp(
            np.flatnonzero(nans), np.flatnonzero(not_nans), shoulder_diff[not_nans]
        )

    # min distance from peak to peak (2 seconds of straight movement) in samples
    min_peak_distance = min_walk_duration * sampling_fr

    peaks, peak_properties = find_peaks(shoulder_diff, height=0.3, distance=min_peak_distance)
    widths, width_heights, left_ips, right_ips = peak_widths(
        x=shoulder_diff, peaks=peaks, rel_height=0.8
    )

    # create mask to indicate straight walking segments (straight=0, turn=1)
    turn_mask = np.zeros_like(shoulder_R)
    for left_base_idx, right_base_idx in zip((left_ips).astype(int), right_ips.astype(int)):
        # turn_segment_length = right_base_idx - left_base_idx
        turn_mask[left_base_idx:right_base_idx] = 1

    # create mask to indicate front vs back perspectives (front=1, back=0)
    perspective = np.where(shoulder_L > shoulder_R, np.True_, np.False_)
    turn_mask = turn_mask.astype(np.bool)
    return (turn_mask, perspective)


def walk_direction_peaks(data: np.ndarray, kpt_labels: list, sampling_fr: float) -> np.ndarray:
    """
    Determines walking state (straight/turning) based on shoulder coordinates using peak detection method.

    Args:
        data:           Input data with shape (n_keypoints, n_dims, n_frames).
        kpt_labels:     List of keypoint labels corresponding to data (left_ankle, nose, etc).
        fps:            sampling frequency

    Returns:
        valid_segments: boolean array indicating straight walking segments (True) and turning segments (False).

    """

    # get the y coord of shoulders (for qualisys x-y plane is horizontal, y coord was forwards/backwards movement)
    shoulder_R = data[kpt_labels.index("right_shoulder"), 1, :]
    shoulder_L = data[kpt_labels.index("left_shoulder"), 1, :]

    shoulder_diff = np.abs(np.diff(shoulder_R - shoulder_L))
    shoulder_diff = shoulder_diff / np.nanmax(shoulder_diff)

    # Interpolate NaNs in shoulder_diff
    if np.any(np.isnan(shoulder_diff)):
        nans = np.isnan(shoulder_diff)
        not_nans = ~nans
        shoulder_diff[nans] = np.interp(
            np.flatnonzero(nans), np.flatnonzero(not_nans), shoulder_diff[not_nans]
        )

    # min distance from peak to peak (2 seconds of straight movement) in samples
    min_peak_distance = 2 * sampling_fr

    peaks, peak_properties = find_peaks(shoulder_diff, height=0.3, distance=min_peak_distance)
    widths, width_heights, left_ips, right_ips = peak_widths(
        x=shoulder_diff, peaks=peaks, rel_height=0.95
    )

    # mark turning segments as invalid for gait analysis
    # valid_segments = np.ones_like(data[0, 0, :], dtype=bool)
    # for left_base, right_base in zip(left_ips, right_ips):
    #     valid_segments[int(np.floor(left_base)) : int(np.ceil(right_base))] = False

    # valid_segments = np.full(shape=data[0, 0, :].shape, fill_value="straight")
    # for left_base, right_base in zip(left_ips, right_ips):
    #     valid_segments[int(np.floor(left_base)) : int(np.ceil(right_base))] = str("turn")

    valid_segments = ["straight"] * data.shape[2]
    for left_base, right_base in zip(left_ips, right_ips):
        turn = ["turn"] * (int(np.floor(right_base)) - int(np.ceil(left_base)) + 2)
        valid_segments[int(np.floor(left_base)) : int(np.ceil(right_base))] = turn

    return valid_segments


def interpolate_gaps(data: np.ndarray, fps: int, max_gap: float) -> np.ndarray:
    """
    Fills gaps below max_gap size along each dimension (1D) of the input array.
    Args:
        data (np.ndarray):  Input data with shape (n_keypoints, n_dims, n_frames).
        fps (int):          Frames per second of the data.
        max_gap (float):    Maximum gap size in seconds


    """
    n_keypoints, n_dims, n_frames = data.shape
    max_gap_frames = int(max_gap * fps)

    filled_data = np.copy(data)  # Preserve original data

    for kpt in range(n_keypoints):
        for dim in range(n_dims):
            signal = data[kpt, dim, :]
            nan_indices = np.where(np.isnan(signal))[0]

            if len(nan_indices) == 0:
                continue

            # Identify NaN segments
            diff = np.diff(nan_indices)
            segment_starts = np.insert(nan_indices[np.where(diff > 1)[0] + 1], 0, nan_indices[0])
            segment_ends = np.append(nan_indices[np.where(diff > 1)[0]], nan_indices[-1])

            # Interpolate gaps within the allowed size
            for start, end in zip(segment_starts, segment_ends):
                gap_size = end - start + 1
                if gap_size <= max_gap_frames:

                    valid_indices = np.where(~np.isnan(signal))[0]
                    # fit cubic spline if there are enough points
                    if len(valid_indices) >= 2:
                        cs = CubicSpline(valid_indices, signal[valid_indices])
                        filled_data[kpt, dim, start : end + 1] = cs(np.arange(start, end + 1))

                    # use linear interpolation if not enough valid points
                    else:
                        non_nan_idx = np.where(~np.isnan(signal))[0]
                        filled_data[kpt, dim, :] = np.interp(
                            np.arange(n_frames), non_nan_idx, signal[non_nan_idx]
                        )

    return filled_data


def filter_data(
    data: np.ndarray,
    sampling_rate: float,
    filter_type: Literal["lowpass", "bandpass"],
    cutoff: list,
    order: int,
    gap_size: int,
) -> np.ndarray:
    """
    Interpolates and applies filter to data (with NaNs).

    Args:
        data (np.ndarray):      Input data with shape (n_kpt, n_dims, n_frames).
        sampling_rate (float):  Sampling rate of the data.
        cutoff (float):         Cutoff frequency for the low-pass filter.
        order (int):            Order of the Butterworth filter.
        gap_size (int):         Maximum size (in seconds) of gaps to fill

    Returns:
        filtered_data (np.ndarray): Interpolated and filtered data with the origianl Nan values in place.

    """
    n_kpt, n_dims, n_frames = data.shape
    data = interpolate_gaps(data, sampling_rate, gap_size)  # Fill short gaps
    filtered_data = copy.deepcopy(data)

    match filter_type:
        case "lowpass":
            cutoff = cutoff[0]
        case "bandpass":
            cutoff = np.asarray(cutoff)
    # Design Butterworth low-pass filter
    b, a = butter(N=order, Wn=cutoff, btype=filter_type, analog=False, fs=sampling_rate)

    for kpt in range(n_kpt):
        for dim in range(n_dims):
            trajectory = data[kpt, dim, :]

            # search for NaNs
            nans = np.isnan(trajectory)

            # if there are nans, interpolate the missing values for subsequent filtering
            if np.any(nans):
                valid_indices = ~nans
                trajectory[nans] = np.interp(
                    np.flatnonzero(nans), np.flatnonzero(valid_indices), trajectory[valid_indices]
                )

            trajectory = filtfilt(b, a, trajectory)  # Apply filter
            trajectory[nans] = np.nan  # Restore NaNs

            filtered_data[kpt, dim, :] = trajectory

    return filtered_data


def step_detection(data: np.ndarray, kpt_labels: list, properties: dict) -> dict:
    """
    asdfasdfasdf
    Args:
        data: filtered pose data (keypoints x dims x frames)
        kpt_labels: list of keypoint labels corresponding to data (left_ankle, nose, etc)
        properties: dictionary of static parameters from properties.json

    Returns:
        events: dictionary matching frame indices to gait events (IC, FC)...
                for each side (left, right) along with perspective (frontal, saggital)
    """

    # perspective = walk_direction(data, kpt_labels, properties["stride_min"], properties["fps"])
    perspective = walk_direction_peaks(data, kpt_labels, properties["fps"])

    events = {"left": {}, "right": {}}

    for side in ["left", "right"]:
        _, _, velocity = bruening_ridge_detection(data, 1, side, kpt_labels, properties)
        ICs, FCs, _ = bruening_ridge_detection(data, velocity, side, kpt_labels, properties)
        events[side]["ICs"] = ICs
        events[side]["FCs"] = FCs

    IC_events = [
        {"frame": frame, "side": side, "perspective": perspective[frame]}
        for side in ["left", "right"]
        for frame in events[side]["ICs"]
    ]
    FC_events = [
        {"frame": frame, "side": side, "perspective": perspective[frame]}
        for side in ["left", "right"]
        for frame in events[side]["FCs"]
    ]

    return ({"IC": IC_events, "FC": FC_events}, perspective)


def walk_direction(
    data: np.ndarray, keypoint_mapping: list, min_length: float, fps: float
) -> np.ndarray:
    """
    Determines walking direction (frontal/saggital) based on shoulder coordinates.
    Args:
        data: interpolated and filtered pose data (keypoints x dims x frames)
        keypoint_mapping: list of keypoint labels corresponding to data (left_ankle, nose, etc)
        min_length: minimum segment length in seconds
        fps: sampling frequency of recorded data

    Returns:
        perspective: array of 'frontal', 'sagittal', or None for each frame
    """

    # get shoulder coordinates (y-axis)
    # note: in qualisys data, y-axis is left-right, x is front-back, z is up-down
    shoulder_R = data[keypoint_mapping.index("right_shoulder"), 1, :]
    shoulder_L = data[keypoint_mapping.index("left_shoulder"), 1, :]

    orientation = shoulder_R - shoulder_L
    perspective = np.where(
        orientation > 0, "frontal", np.where(orientation < 0, "sagittal", None)
    ).astype(object)

    # max gap size in frames
    max_gap_size = int(min_length * fps)
    valid_indices = np.where(perspective != None)[0]

    # TODO: figure out what this does, add comments
    for start, end in zip(valid_indices[:-1], valid_indices[1:]):
        if end - start <= max_gap_size:
            perspective[start + 1 : end] = perspective[start]

    changes = np.r_[True, perspective[:-1] != perspective[1:], True]
    segment_starts, segment_ends = np.where(changes[:-1])[0], np.where(changes[1:])[0] - 1

    for start, end in zip(segment_starts, segment_ends):
        if (end - start + 1) / fps < min_length:
            perspective[start : end + 1] = None

    return perspective


def bruening_ridge_detection(
    data: np.ndarray, velocity: float, side: str, kpt_labels: list, properties: dict
) -> tuple:
    """
    Identifies Gait Events (GE) using the velocity of markers on the foot (heel, toe, ankle).

    Args:
        data: interpolated and filtered pose data from qualisys (keypoints x dims x frames)
        velocity: initial walking velocity estimate (m/s)
        side: 'left' or 'right'
        kpt_labels: list of keypoint labels corresponding to data (left_ankle, nose, etc)
        properties: dictionary of static parameters from properties.json

    Returns:

        ICs: list of Initial Contact (IC) frame indices

        FCs: list of Final Contact (FC) frame indices

        velocity: computed walking velocity (m/s)

    """
    fs = properties["fps"]
    heel_thr = properties["heel_thr"] * velocity
    # use ankle marker in case the other 2 are not visible
    ankle_thr = properties["heel_thr"] * velocity
    big_toe_thr = properties["toe_thr"] * velocity

    # Extract trajectories
    heel = data[kpt_labels.index(f"{side}_heel"), :, :]
    big_toe = data[kpt_labels.index(f"{side}_big_toe"), :, :]
    ankle = data[kpt_labels.index(f"{side}_ankle"), :, :]

    # Compute 3D velocities
    heel_vel = np.linalg.norm(np.diff(heel, axis=1), axis=0) * fs
    ankle_vel = np.linalg.norm(np.diff(ankle, axis=1), axis=0) * fs
    big_toe_vel = np.linalg.norm(np.diff(big_toe, axis=1), axis=0) * fs

    # Ground contact detection based on thresholds
    ground_contact = (
        (heel_vel < heel_thr) | (ankle_vel < ankle_thr) | (big_toe_vel < big_toe_thr)
    ).astype(int)

    # Remove short ground contact periods
    min_gc_duration = int(properties["stance_min"] * fs)
    min_no_gc_duration = int(properties["swing_min"] * fs)

    contact_diff = np.diff(np.r_[0, ground_contact, 0])
    starts = np.where(contact_diff == 1)[0]
    ends = np.where(contact_diff == -1)[0]

    for start, end in zip(starts, ends):
        if end - start < min_gc_duration:
            ground_contact[start:end] = 0

    # Remove short no-contact periods
    contact_diff = np.diff(np.r_[0, ground_contact, 0])
    starts = np.where(contact_diff == 1)[0]
    ends = np.where(contact_diff == -1)[0]

    for start, end in zip(starts, ends):
        if end - start < min_no_gc_duration:
            ground_contact[start:end] = 1

    # Identify initial contacts (ICs) and final contacts (FCs)
    ground_contact_diff = np.diff(ground_contact)
    ICs = np.where(ground_contact_diff == 1)[0] + 1
    FCs = np.where(ground_contact_diff == -1)[0] + 1

    # Compute walking velocity from stride lengths and durations
    stride_lengths = []
    stride_durations = []

    for i in range(1, len(ICs)):
        stride_duration = (ICs[i] - ICs[i - 1]) / fs
        if properties["stride_min"] <= stride_duration <= properties["stride_max"]:
            stride_length = np.linalg.norm(heel[:, ICs[i]] - heel[:, ICs[i - 1]])
            stride_lengths.append(stride_length)
            stride_durations.append(stride_duration)

    velocity = (
        np.nanmean(np.array(stride_lengths) / np.array(stride_durations))
        if stride_durations
        else np.nan
    )

    return ICs, FCs, velocity


def get_frame_index(gait_events: list, side: str, lower_bound: int, upper_bound: int) -> list:
    """
    Return frame indices from gait_events with the given side from lower_bound to upper_bound frame index.

    Args:
        gait_events: input data
        side: 'left' or 'right'
        lower_bound: lower frame index
        upper_bound: upper frame index
    Returns:
        list of gait events
    """
    return [
        event["frame"]
        for event in gait_events
        if event["side"] == side and lower_bound < event["frame"] < upper_bound
    ]


def compute_pooled_stats(left_values, right_values):
    """
    Calculates
    """
    pooled = np.concatenate([left_values, right_values])
    pooled = pooled[~np.isnan(pooled)]
    mean = np.mean(pooled) if len(pooled) > 0 else np.nan
    # variabilty (std normalized by mean) (coeff of variation)
    cv = 100 * np.std(pooled) / mean if mean != 0 else np.nan
    return {"mean": mean, "CV": cv}


def compute_asymmetry(left_values, right_values):
    """
    Calculates asymmetry between left and right sides
    """
    left, right = map(lambda x: np.array(x)[~np.isnan(x)], [left_values, right_values])
    if len(left) == 0 or len(right) == 0:
        return np.nan
    larger, smaller = max(left.mean(), right.mean()), min(left.mean(), right.mean())
    return 100 * (1 - smaller / larger) if larger > 0 else np.nan


def gait_analysis(data: np.ndarray, events: dict, keypoint_mapping: list, properties: dict) -> dict:

    # static properties for calculations
    fs = properties["fps"]
    stride_min = properties["stride_min"]
    stride_max = properties["stride_max"]

    perspectives = ["all", "straight", "turn"]
    # metrics of interest (for each side)
    moi = {
        metric: []
        for metric in [
            "stime",
            "slen",
            "vel",
            "swing",
            "dsupp",
            "bos",
        ]
    }
    # output data structure
    metrics = {
        perspective: {
            "left": copy.deepcopy(moi),
            "right": copy.deepcopy(moi),
        }
        for perspective in perspectives
    }

    # Initial- and Final-contac gait events
    ICs = events["IC"]
    FCs = events["FC"]

    # iterate Initial Contact gait events
    for i, IC in enumerate(ICs):

        # ipsilateral and contralateral sides for current gait event
        ipsi, contra = IC["side"], "left" if IC["side"] == "right" else "right"

        # perspective value ('straight'/'turn') or 'all' if missing
        perspective = IC.get("perspective", "all")

        if perspective not in perspectives:
            continue

        # get the next ipsilateral event's global index (relative to the whole event list), return None otherwise
        same_foot_next_idx = next(
            (j for j, event in enumerate(ICs[i + 1 :], start=i + 1) if event["side"] == ipsi), None
        )

        # check if event order is correct, filter false positives
        if same_foot_next_idx is not None:

            # stride time = time elapsed between heelstrikes of the same foot
            stime = (ICs[same_foot_next_idx]["frame"] - IC["frame"]) / fs

            # stride time falls in realistic time range
            if stride_min <= stime <= stride_max:
                # frame index of current and next IC event (ipsilateral)
                IC0, IC2 = IC["frame"], ICs[same_foot_next_idx]["frame"]

                # frame index of first contralateral heel strike (between current and next ipsilateral)
                IC1 = get_frame_index(ICs, contra, IC0, IC2)

                FC0, FC1, FC2 = None, None, None

                # if there's a next contralateral GE
                if IC1:
                    IC1 = IC1[0]

                    # see gait_events.png
                    FC0 = get_frame_index(FCs, contra, IC0, IC1)
                    FC1 = get_frame_index(FCs, ipsi, IC1, IC2)
                    FC2 = get_frame_index(FCs, ipsi, IC2, IC2 + int(fs * stride_max))

                    FC0 = FC0[0] if FC0 else None
                    FC1 = FC1[0] if FC1 else None
                    FC2 = FC2[0] if FC2 else None

                # if either of the values is None, ignore it
                if any(x is None for x in [IC0, IC1, IC2, FC0, FC1, FC2]):
                    continue

                # heel point
                HP0 = np.nanmedian(data[keypoint_mapping.index(f"{ipsi}_heel"), :, IC0:FC0], axis=1)
                HP2 = np.nanmedian(data[keypoint_mapping.index(f"{ipsi}_heel"), :, IC2:FC2], axis=1)
                HP1 = np.nanmedian(
                    data[keypoint_mapping.index(f"{contra}_heel"), :, IC1:FC1], axis=1
                )

                # stride lenght on xy plane
                slen = np.linalg.norm(HP2[:2] - HP0[:2])
                vel = slen / stime
                bos = np.linalg.norm(np.cross(HP2 - HP1, HP1 - HP0)) / np.linalg.norm(HP2 - HP1)
                swing = (IC2 - FC1) / fs
                dsupp = ((FC0 - IC0) + (FC1 - IC1)) / fs

                for pers in ["all", perspective]:
                    metrics[pers][ipsi]["stime"].append(stime)
                    metrics[pers][ipsi]["slen"].append(slen)
                    metrics[pers][ipsi]["vel"].append(vel)
                    metrics[pers][ipsi]["swing"].append(swing)
                    metrics[pers][ipsi]["dsupp"].append(dsupp)
                    metrics[pers][ipsi]["bos"].append(bos)

    parameters = {}
    for state, data in metrics.items():
        parameters[state] = {
            metric: {
                **compute_pooled_stats(data["left"][metric], data["right"][metric]),
                "asymmetry": compute_asymmetry(data["left"][metric], data["right"][metric]),
            }
            for metric in data["left"]
        }

    return parameters


def display_results(parameters):
    parameter_order = [
        "stime",
        "slen",
        "vel",
        "swing",
        "dsupp",
        "bos",
    ]
    statistic_order = ["Mean", "CV", "Asymmetry"]

    rows = []
    for segment_type, metrics in parameters.items():
        for param, values in metrics.items():
            rows.extend(
                [
                    {
                        "Parameter": param,
                        "Statistic": "Mean",
                        "Perspective": segment_type,
                        "Value": values["mean"],
                    },
                    {
                        "Parameter": param,
                        "Statistic": "CV",
                        "Perspective": segment_type,
                        "Value": values["CV"],
                    },
                    {
                        "Parameter": param,
                        "Statistic": "Asymmetry",
                        "Perspective": segment_type,
                        "Value": values["asymmetry"],
                    },
                ]
            )
    pd.options.display.float_format = "{:,.1f}".format
    df = pd.DataFrame(rows)
    df["Parameter"] = pd.Categorical(df["Parameter"], categories=parameter_order, ordered=True)
    df["Statistic"] = pd.Categorical(df["Statistic"], categories=statistic_order, ordered=True)
    df = df.sort_values(by=["Parameter", "Statistic"])

    table = df.pivot_table(
        index=["Parameter", "Statistic"],
        columns="Perspective",
        values="Value",
        aggfunc="mean",
        observed=False,
    )
    table = table.reset_index()
    table.columns.name = None

    print(table)
    return table


def filter_2d_keypoint(
    keypoint: np.array, order=2, lowcut=1, fr=60, create_diagram=False, title="asdf"
) -> np.array:
    """
    Perfortm FFT and low-pass filtering on a single 2D keypoint (e.g.: left elbow, (N, (x,y,conf))).

    Returns the filetered keypoint coordinates (and confidence) in the original shape (N, 3).
    """

    # x and y coordinates
    kp_x = keypoint[:, 0]
    kp_y = keypoint[:, 1]
    # conf = keypoint[:, 2]

    filtered_keypoint = np.empty_like(keypoint)
    filtered_x, filtered_y = (
        np.empty_like(kp_x),
        np.empty_like(kp_y),
    )

    # time
    t = np.arange(0, keypoint.shape[0], 1) / fr

    # filtering parameters
    b, a = butter(order, lowcut, fs=fr, btype="low", analog=False)

    for coords in [kp_x, kp_y]:

        original = coords.copy()
        nan_indices = np.isnan(coords)
        valid_indices = ~nan_indices
        # replace NaN values with zeros
        if nan_indices.any():
            print("NaN values found in keypoint data. Replacing with mean.")
            # mean = np.nanmean(coords)
            coords[nan_indices] = np.interp(
                np.flatnonzero(nan_indices), np.flatnonzero(valid_indices), coords[valid_indices]
            )

        # filtered x or y coords
        filtered_coord = sp_signal.filtfilt(b, a, coords)

        # put nans back in
        if nan_indices.any():
            filtered_coord[nan_indices] = np.nan

        if np.array_equiv(coords, kp_x):
            filtered_x = filtered_coord
            subtitle = "x coordinate"
            yl = 0.1

        elif np.array_equiv(coords, kp_y):
            filtered_y = filtered_coord
            subtitle = "y coordinate"
            yl = 0.005
        # elif np.array_equiv(coords, kp_d):
        #     filtered_d = filtered_coord
        #     subtitle = "depth"
        #     yl = 0.1

        if create_diagram:
            plt.figure(figsize=(15, 3))
            plt.title(subtitle)

            plt.suptitle(f"FFT of {title} Keypoint", fontsize=16)
            colors = ["r", "b"]

            # perform FFT on each coordinate(x or y), plot results
            for i, signal in enumerate([filtered_coord, original]):

                X = fft(signal)
                N = len(X)
                n = np.arange(N)
                T = N / 60
                freq = n / T
                plt.subplot(1, 2, 1)
                plt.plot(
                    freq,
                    np.abs(X),
                    colors[i],
                    label=f'{title} {["Filtered", "Original"][i]}',
                    alpha=[1, 0.5][i],
                )
                plt.tight_layout()

            plt.xlabel("Freq (Hz)")
            plt.ylabel("FFT Amplitude |X(freq)|")
            plt.xlim(0, fr / 2)
            ylim = max(np.abs(X) * yl)
            # plt.ylim(0, ylim)
            plt.legend(["Filtered", "Original"])

            # plot the original and iFFT signals
            plt.subplot(122)
            plt.plot(t, ifft(X), "r")
            plt.plot(t, original, "b--")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude [px]")
            plt.legend(["iFFT", "Original"])
            plt.tight_layout()
            plt.show()

            # plot the original and filtered signals
            plt.figure(figsize=(15, 3))
            plt.plot(t, original, "b--", label="Original")
            plt.plot(t, filtered_coord, "r", label="Filtered")
            # if nan_indices.any():
            #     plt.plot(t[nan_indices],filtered_coord[nan_indices], "g*", label='Nans')
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude [px]")
            plt.legend()
            plt.grid()
            plt.show()

    # stack filtered coords, use original confidence
    # filtered_keypoint = np.stack([filtered_x, filtered_y, keypoint[:, 2]], axis=1)
    filtered_keypoint = np.stack([filtered_x, filtered_y], axis=1)

    return filtered_keypoint


def progress_bar(percent_done, bar_length=50):
    # Display a progress bar
    done_length = int(bar_length * percent_done / 100)
    bar = "=" * done_length + "-" * (bar_length - done_length)
    sys.stdout.write("[%s] %i%s\r" % (bar, percent_done, "%"))
    sys.stdout.flush()


class BodyWithFeet:
    """
    Halpe26 class for human pose estimation using the Halpe26 keypoint format.
    This class supports different modes of operation and can output in OpenPose format.
    """

    MODE = {
        "performance": {
            "det": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip",
            "det_input_size": (640, 640),
            "pose": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-x_simcc-body7_pt-body7-halpe26_700e-384x288-7fb6e239_20230606.zip",
            "pose_input_size": (288, 384),
        },
        "lightweight": {
            "det": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_tiny_8xb8-300e_humanart-6f3252f9.zip",
            "det_input_size": (416, 416),
            "pose": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-s_simcc-body7_pt-body7-halpe26_700e-256x192-7f134165_20230605.zip",
            "pose_input_size": (192, 256),
        },
        "balanced": {
            "det": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip",
            "det_input_size": (640, 640),
            "pose": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.zip",
            "pose_input_size": (192, 256),
        },
    }

    def __init__(
        self,
        det: str = None,
        det_input_size: tuple = (640, 640),
        pose: str = None,
        pose_input_size: tuple = (192, 256),
        mode: str = "balanced",
        to_openpose: bool = False,
        backend: str = "onnxruntime",
        device: str = "cpu",
    ):
        """
        Initialize the Halpe26 pose estimation model.

        Args:
            det (str, optional): Path to detection model. If None, uses default based on mode.
            det_input_size (tuple, optional): Input size for detection model. Default is (640, 640).
            pose (str, optional): Path to pose estimation model. If None, uses default based on mode.
            pose_input_size (tuple, optional): Input size for pose model. Default is (192, 256).
            mode (str, optional): Operation mode ('performance', 'lightweight', or 'balanced'). Default is 'balanced'.
            to_openpose (bool, optional): Whether to convert output to OpenPose format. Default is False.
            backend (str, optional): Backend for inference ('onnxruntime' or 'opencv'). Default is 'onnxruntime'.
            device (str, optional): Device for inference ('cpu' or 'cuda'). Default is 'cpu'.
        """
        from rtmlib import YOLOX, RTMPose

        if pose is None:
            pose = self.MODE[mode]["pose"]
            pose_input_size = self.MODE[mode]["pose_input_size"]

        if det is None:
            det = self.MODE[mode]["det"]
            det_input_size = self.MODE[mode]["det_input_size"]

        self.det_model = YOLOX(det, model_input_size=det_input_size, backend=backend, device=device)
        self.pose_model = RTMPose(
            pose,
            model_input_size=pose_input_size,
            to_openpose=to_openpose,
            backend=backend,
            device=device,
        )

    def __call__(self, image: np.ndarray):
        """
        Perform pose estimation on the input image.

        Args:
            image (np.ndarray): Input image for pose estimation.

        Returns:
            tuple: A tuple containing:
                - keypoints (np.ndarray): Estimated keypoint coordinates.
                - scores (np.ndarray): Confidence scores for each keypoint.
        """
        bboxes = self.det_model(image)
        keypoints, scores = self.pose_model(image, bboxes=bboxes)
        return keypoints, scores


def compute_iou(bboxA: list, bboxB: list) -> float:
    """
    Compute the Intersection over Union (IoU) between two boxes.
    (How much two bboxes overlap relative to their combined area)

    Args:
        bboxA (list): The first bbox info (left, top, right, bottom, score).
        bboxB (list): The second bbox info (left, top, right, bottom, score).

    Returns:
        float: The IoU value.
    """

    # find the corners of the intersection area
    x1 = max(bboxA[0], bboxB[0])
    y1 = max(bboxA[1], bboxB[1])
    x2 = min(bboxA[2], bboxB[2])
    y2 = min(bboxA[3], bboxB[3])

    # calculate area of intersection
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    # area of bbox A & B
    bboxA_area = (bboxA[2] - bboxA[0]) * (bboxA[3] - bboxA[1])
    bboxB_area = (bboxB[2] - bboxB[0]) * (bboxB[3] - bboxB[1])

    # area of union
    union_area = float(bboxA_area + bboxB_area - inter_area)
    if union_area == 0:
        union_area = 1e-5
        warnings.warn("union_area=0 is unexpected")

    iou = inter_area / union_area

    return iou


def pose_to_bbox(keypoints: np.ndarray, expansion: float = 1.25) -> np.ndarray:
    """Get bounding box from keypoints.

    Args:
        keypoints (np.ndarray): Keypoints of person.
        expansion (float): Expansion ratio of bounding box.

    Returns:
        np.ndarray: Bounding box of person.
    """
    x = keypoints[:, 0]
    y = keypoints[:, 1]
    bbox = np.array([x.min(), y.min(), x.max(), y.max()])
    center = np.array([bbox[0] + bbox[2], bbox[1] + bbox[3]]) / 2
    bbox = np.concatenate(
        [
            center - (center - bbox[:2]) * expansion,
            center + (bbox[2:] - center) * expansion,
        ]
    )
    return bbox


class PoseTracker:
    """
    Pose tracker for pose estimation.

    Args:
        solution (type): rtmlib solutions, e.g. Wholebody, Body, Custom, etc.
        det_frequency (int): Frequency of object detection (performed every det_frequency number of frames).
        mode (str): 'performance', 'lightweight', or 'balanced'.
        to_openpose (bool): Whether to use openpose-style skeleton.
        backend (str): Backend of pose estimation model.
        device (str): Device of pose estimation model.

    Returns:
        faszom nem is mukodik
    """

    MIN_AREA = 100

    def __init__(
        self,
        solution: type,
        det_frequency: int = 1,
        tracking: bool = True,
        tracking_thr: float = 0.3,
        mode: str = "balanced",
        to_openpose: bool = False,
        backend: str = "onnxruntime",
        device: str = "gpu",
    ):

        # load the predefined solution (detection model, estimation model, backend...)
        model = solution(mode=mode, to_openpose=to_openpose, backend=backend, device=device)

        try:
            self.det_model = model.det_model
        except:  # rtmo
            self.det_model = None
        self.pose_model = model.pose_model

        self.det_frequency = det_frequency
        self.tracking = tracking
        self.tracking_thr = tracking_thr
        self.reset()

        if self.tracking:
            print(
                "Tracking is on, you can get higher FPS by turning it off:"
                "`PoseTracker(tracking=False)`"
            )

    def reset(self):
        """Reset pose tracker."""
        self.frame_cnt = 0
        self.next_id = 0
        self.bboxes_last_frame = []
        self.track_ids_last_frame = []

    def __call__(self, image: np.ndarray):

        # if a detection model is used
        if self.det_model is not None:
            # if it is the 1st frame, use bboxes from the frame
            if self.frame_cnt % self.det_frequency == 0:
                bboxes = self.det_model(image)
            # not 1st frame,
            else:
                bboxes = self.bboxes_last_frame
            keypoints, scores = self.pose_model(image, bboxes=bboxes)

        # no detection model is used, use just pose estimator
        else:  # rtmo
            keypoints, scores = self.pose_model(image)

        if not self.tracking:
            # without tracking
            bboxes_current_frame = []
            for kpts in keypoints:
                bbox = pose_to_bbox(kpts)
                bboxes_current_frame.append(bbox)

        else:
            # with tracking
            if len(self.track_ids_last_frame) == 0:
                self.next_id = len(self.bboxes_last_frame)
                self.track_ids_last_frame = list(range(self.next_id))

            bboxes_current_frame = []
            track_ids_current_frame = []

            for kpts in keypoints:
                bbox = pose_to_bbox(kpts)

                track_id, _ = self.track_by_iou(bbox)

                if track_id > -1:
                    track_ids_current_frame.append(track_id)
                    bboxes_current_frame.append(bbox)

            self.track_ids_last_frame = track_ids_current_frame

            # reorder keypoints, scores according to track_id
            # keypoints = np.array([keypoints[i] for i in self.track_ids_last_frame])
            # scores = np.array([scores[i] for i in self.track_ids_last_frame])

        self.bboxes_last_frame = bboxes_current_frame
        self.frame_cnt += 1

        return keypoints, scores, track_ids_current_frame, self.track_ids_last_frame

    def track_by_iou(self, bbox):
        """Get track id using IoU tracking greedily.

        Args:
            bbox (list): The bbox info (left, top, right, bottom, score).
            next_id (int): The next track id.

        Returns:
            track_id (int): The track id.
            match_result (list): The matched bbox.
            next_id (int): The updated next track id.
        """

        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

        max_iou_score = -1
        max_index = -1
        match_result = None

        # iterate bboxes of last frame
        for index, each_bbox in enumerate(self.bboxes_last_frame):

            iou_score = compute_iou(bbox, each_bbox)
            if iou_score > max_iou_score:
                max_iou_score = iou_score
                max_index = index

        if max_iou_score > self.tracking_thr:
            # if the bbox has a match and the IoU is larger than threshold
            track_id = self.track_ids_last_frame.pop(max_index)
            match_result = self.bboxes_last_frame.pop(max_index)

        elif area >= self.MIN_AREA:
            # no match, but the bbox is large enough,
            # assign a new track id
            track_id = self.next_id
            self.next_id += 1

        else:
            # if the bbox is too small, ignore it
            track_id = -1

        return track_id, match_result


class Custom:

    def __init__(
        self,
        det_class: str = None,
        det: str = None,
        det_input_size: tuple = (640, 640),
        pose_class: str = None,
        pose: str = None,
        pose_input_size: tuple = (192, 256),
        mode: str = None,
        to_openpose: bool = False,
        backend: str = "onnxruntime",
        device: str = "cuda",
    ):

        if det_class is not None:
            try:
                det_class = getattr(rtmlib_module, det_class)
                self.det_model = det_class(
                    det, model_input_size=det_input_size, backend=backend, device=device
                )
                self.one_stage = False

            except ImportError:
                raise ImportError(f"{det_class} is not supported by rtmlib.")
        else:
            self.one_stage = True

        if pose_class is not None:
            try:
                pose_class = getattr(rtmlib_module, pose_class)
                self.pose_model = pose_class(
                    pose,
                    model_input_size=pose_input_size,
                    to_openpose=to_openpose,
                    backend=backend,
                    device=device,
                )
            except ImportError:
                raise ImportError(f"{pose_class} is not supported by rtmlib.")

    def __call__(self, image: np.ndarray):
        if self.one_stage:
            keypoints, scores = self.pose_model(image)
        else:
            bboxes = self.det_model(image)
            keypoints, scores = self.pose_model(image, bboxes=bboxes)

        return keypoints, scores


class Body:
    MODE = {
        "performance": {
            "det": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip",  # noqa
            "det_input_size": (640, 640),
            "pose": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629.zip",  # noqa
            "pose_input_size": (288, 384),
        },
        "lightweight": {
            "det": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_tiny_8xb8-300e_humanart-6f3252f9.zip",  # noqa
            "det_input_size": (416, 416),
            "pose": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.zip",  # noqa
            "pose_input_size": (192, 256),
        },
        "balanced": {
            "det": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip",  # noqa
            "det_input_size": (640, 640),
            "pose": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip",  # noqa
            "pose_input_size": (192, 256),
        },
    }

    RTMO_MODE = {
        "performance": {
            "pose": "https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.zip",  # noqa
            "pose_input_size": (640, 640),
        },
        "lightweight": {
            "pose": "https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-s_8xb32-600e_body7-640x640-dac2bf74_20231211.zip",  # noqa
            "pose_input_size": (640, 640),
        },
        "balanced": {
            "pose": "https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.zip",  # noqa
            "pose_input_size": (640, 640),
        },
    }

    def __init__(
        self,
        det: str = None,
        det_input_size: tuple = (640, 640),
        pose: str = None,
        pose_input_size: tuple = (288, 384),
        mode: str = "balanced",
        to_openpose: bool = False,
        backend: str = "onnxruntime",
        device: str = "cpu",
    ):

        if pose is not None and "rtmo" in pose:
            from rtmlib import RTMO

            self.one_stage = True

            pose = self.RTMO_MODE[mode]["pose"]
            pose_input_size = self.RTMO_MODE[mode]["pose_input_size"]
            self.pose_model = RTMO(
                pose,
                model_input_size=pose_input_size,
                to_openpose=to_openpose,
                backend=backend,
                device=device,
            )
        else:
            from rtmlib import YOLOX, RTMPose

            self.one_stage = False

            if pose is None:
                pose = self.MODE[mode]["pose"]
                pose_input_size = self.MODE[mode]["pose_input_size"]

            if det is None:
                det = self.MODE[mode]["det"]
                det_input_size = self.MODE[mode]["det_input_size"]

            self.det_model = YOLOX(
                det, model_input_size=det_input_size, backend=backend, device=device
            )
            self.pose_model = RTMPose(
                pose,
                model_input_size=pose_input_size,
                to_openpose=to_openpose,
                backend=backend,
                device=device,
            )

    def __call__(self, image: np.ndarray):
        if self.one_stage:
            keypoints, scores = self.pose_model(image)
        else:
            bboxes = self.det_model(image)
            keypoints, scores = self.pose_model(image, bboxes=bboxes)

        return keypoints, scores
