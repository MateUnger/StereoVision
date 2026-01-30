import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, peak_widths

import copy
from typing import Literal

#-----------------------------------------STEP DETECTION---------------------------------------------

def get_turns_and_perspective(
    keypoint_data: np.ndarray,
    kpt_labels: list,
    sampling_fr: float,
    min_peak_height: float = 0.3,
    relative_height: float = 0.85,
    min_walk_duration: float = 2,
    debug_fig_file_path: str = None,
) -> tuple:
    """
    Identifies turning segments (straight/turning) and perspective (front / back) based on shoulder coordinates using peak detection method.

    Args:
        data:               Input data with shape (n_keypoints, n_dims, n_frames).
        kpt_labels:         List of keypoint labels corresponding to data (left_ankle, nose, etc).
        sampling_fr:        sampling frequency
        min_peak_height:    only peaks above this height will be considered
        relative_height:    height at which the bases of a peak will be taken. It is MEASURED FROM APEX
        min_walk_duration:  min elapsed time (in seconds) between turns (1 turn = 180 deg)

    Returns:
        turn_mask: boolean array indicating straight turning segments (1) and straight segments (0).

        perspective: bollean array indicating frontal facing (1) and back facing (0) segments.
    """
    # get the y coord of shoulders (for qualisys x-y plane is horizontal, y coord was forwards/backwards movement)
    shoulder_R = keypoint_data[kpt_labels.index("right_shoulder"), 1, :]
    shoulder_L = keypoint_data[kpt_labels.index("left_shoulder"), 1, :]

    # normalize signal
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

    peaks, peak_properties = find_peaks(
        shoulder_diff, height=min_peak_height, distance=min_peak_distance
    )
    widths, width_heights, left_ips, right_ips = peak_widths(
        x=shoulder_diff, peaks=peaks, rel_height=relative_height
    )

    # create mask to indicate straight walking segments (straight=False, turn=True)
    turn_mask = np.zeros_like(shoulder_R)
    for left_base_idx, right_base_idx in zip(
        (left_ips).astype(int), np.ceil(right_ips).astype(int)
    ):
        # turn_segment_length = right_base_idx - left_base_idx
        turn_mask[left_base_idx:right_base_idx] = 1

    # create mask to indicate front vs back perspectives (front=True, back=False)
    perspective = np.where(shoulder_L > shoulder_R, np.True_, np.False_)
    turn_mask = turn_mask.astype(np.bool)

    if debug_fig_file_path is not None:

        # basename, extension = os.path.splitext(os.path.basename(debug_fig_file_path))
        # save_folder = "debug_figs"
        # filename = os.path.join(save_folder, f"{basename}_turns_and_perspective")
        filename = f"{os.path.splitext(debug_fig_file_path)[0]}_turns_and_perspective"

        t = np.linspace(0, len(shoulder_R) / sampling_fr, len(shoulder_R))

        plt.close("all")
        fig, axs = plt.subplots(2, 1, figsize=(14, 4))

        axs[0].plot(t[1:], shoulder_diff, label="shoulder diff")
        axs[0].plot(t, perspective, label="front, back")
        axs[0].plot(
            left_ips / sampling_fr, width_heights, "o", label="turn start", markersize=4
        )
        axs[0].plot(
            right_ips / sampling_fr, width_heights, "o", label="turn end", markersize=4
        )
        axs[0].plot(t, turn_mask, label="straight/turn")

        axs[1].plot(t, shoulder_L, label="left")
        axs[1].plot(t, shoulder_R, label="right")

        for ax in axs:
            ax.legend(loc="upper left")
            ax.grid()

        plt.tight_layout()
        plt.savefig(filename)
        plt.close("all")
        print(f"figure saved to: \n{filename}")

    return (turn_mask, perspective)


def get_gait_events_one_side(
    keypoint_data: np.ndarray,
    kpt_labels: list,
    side: Literal["left", "right"],
    gait_analysis_properties: dict,
    velocity: float,
    turn_mask: np.ndarray,
    perspective: np.ndarray,
    debug_fig_file_path: str = None,
) -> tuple:
    """
    Identifies Gait Events (GE) using the velocity of markers on the foot (heel, toe, ankle).
    Also calculates the velocity profile to iteratively update GEs (2 stage).

    Args:
        data: interpolated and filtered keypoint data (keypoints x dims x frames)
        velocity: initial walking velocity estimate (m/s)
        side: 'left' or 'right'
        kpt_labels: list of keypoint labels corresponding to data (left_ankle, nose, etc)
        gait_analysis_properties: dictionary of static parameters from gait_analysis_properties.json

    Returns:
        ICs: list of Initial Contact (IC) frame indices
        FCs: list of Final Contact (FC) frame indices
        velocity: computed walking velocity (m/s)
    """

    fs = gait_analysis_properties["fps"]
    heel_thr = gait_analysis_properties["heel_thr"] * velocity
    # use ankle marker too (in case the other 2 are not visible, also better for stereo - no obstuction issues)
    ankle_thr = gait_analysis_properties["heel_thr"] * velocity
    big_toe_thr = gait_analysis_properties["toe_thr"] * velocity

    # Extract trajectories
    heel = keypoint_data[kpt_labels.index(f"{side}_heel"), :, :]
    big_toe = keypoint_data[kpt_labels.index(f"{side}_big_toe"), :, :]
    ankle = keypoint_data[kpt_labels.index(f"{side}_ankle"), :, :]

    # Compute 3D velocity magnitudes
    heel_vel = np.linalg.norm(np.diff(heel, axis=1), axis=0) * fs
    ankle_vel = np.linalg.norm(np.diff(ankle, axis=1), axis=0) * fs
    big_toe_vel = np.linalg.norm(np.diff(big_toe, axis=1), axis=0) * fs

    # Ground contact detection based on thresholds
    ground_contact = (
        (heel_vel < heel_thr) | (ankle_vel < ankle_thr) | (big_toe_vel < big_toe_thr)
    ).astype(int)

    min_gc_duration = int(gait_analysis_properties["stance_min"] * fs)
    min_no_gc_duration = int(gait_analysis_properties["swing_min"] * fs)

    # Remove too short ground contact periods
    contact_diff = np.diff(
        np.pad(ground_contact, 1, "constant")
    )  # NOTE:pad beginning and end of array with 0 for diff
    starts = np.nonzero(contact_diff == 1)[0]
    ends = np.nonzero(contact_diff == -1)[0]

    for start, end in zip(starts, ends):
        if end - start < min_gc_duration:
            ground_contact[start:end] = 0

    # Remove too short swing periods
    contact_diff = np.diff(np.pad(ground_contact, 1, "constant"))
    starts = np.nonzero(contact_diff == 1)[0]
    ends = np.nonzero(contact_diff == -1)[0]

    for start, end in zip(starts, ends):
        if end - start < min_no_gc_duration:
            ground_contact[start:end] = 1

    # exclude gait events of turn segments
    ground_contact_diff = np.pad(np.diff(ground_contact), 1, "constant")
    ground_contact_diff_straight = np.where(turn_mask, 0, ground_contact_diff)

    # Identify initial contacts (ICs) and final contacts (FCs)
    ICs = np.nonzero(ground_contact_diff_straight == 1)[0] + 1
    FCs = np.nonzero(ground_contact_diff_straight == -1)[0] + 1

    # Compute walking velocity from stride lengths and durations
    # also filter false positive gait events based on them
    stride_lengths = []
    stride_durations = []
    bad_GE_indices = []

    direction = []

    # refine ICs & Fcs based on stride duration and perspective
    for i in range(len(ICs) - 1):
        current_IC = ICs[i]
        next_IC = ICs[i + 1]

        # temporal difference between current & next IC event
        stride_duration = (next_IC - current_IC) / fs

        # check if gait events belong to same straight segment
        facing_same_way = perspective[current_IC] == perspective[next_IC]
        stride_duration_ok = (
            gait_analysis_properties["stride_min"]
            <= stride_duration
            <= gait_analysis_properties["stride_max"]
        )

        if not stride_duration_ok and facing_same_way:
            bad_GE_indices.append(i + 1)

        if stride_duration_ok and facing_same_way:
            # take the spatial difference of heel keypoints between current & next IC event
            stride_length_heel = np.linalg.norm(heel[:, next_IC] - heel[:, current_IC])
            stride_lengths.append(stride_length_heel)

            stride_durations.append(stride_duration)
            direction.append(perspective[current_IC])

    # remove false positive gait events
    ICs = np.delete(ICs, bad_GE_indices)
    FCs = np.delete(FCs, bad_GE_indices)

    stride_lengths = np.array(stride_lengths)
    stride_durations = np.array(stride_durations)

    mean_velocity = np.nanmean(stride_lengths / stride_durations)

    create_debug_fig = False
    if debug_fig_file_path != None:
        create_debug_fig = True

    # fmt: off
    if create_debug_fig:

        # basename, extension = os.path.splitext(os.path.basename(debug_fig_file_path))
        # save_folder = "debug_figs"
        # filename = os.path.join(save_folder, f"{basename}_gait_events_{side}")
        filename = f"{os.path.splitext(debug_fig_file_path)[0]}_{side}"

        # create array to visualize gait events after filtering out bad ones
        gait_event_vis = np.zeros_like(ground_contact_diff_straight)
        gait_event_vis[ICs] = 1
        gait_event_vis[FCs] = -1
        tS = np.linspace(0, len(ankle_vel) / fs, len(ankle_vel))

        plt.close("all")
        fig, axs = plt.subplots(5, 1, figsize=(14, 9))

        axs[0].plot(tS, ankle_vel, label="velocity")
        axs[0].plot(tS, ground_contact * ankle_thr, label="ground contact at thr")

        axs[1].plot(tS, big_toe_vel, label="velocity")
        axs[1].plot(tS, ground_contact * big_toe_thr, label="ground contact at thr")

        axs[2].plot(tS, heel_vel, label="velocity")
        axs[2].plot(tS, ground_contact * heel_thr, label="ground contact at thr")

        axs[3].plot(tS, ground_contact_diff[1:], alpha=0.4, label="ground contact diff")
        axs[3].plot(tS[1:], turn_mask[2:], "r--", label="turn mask")
        axs[3].plot(tS, gait_event_vis[1:], "b", alpha=1, label="gc diff straight")
        axs[3].plot(tS, ground_contact_diff_straight[1:], "r", alpha=0.4, label="removed GEs")

        axs[4].plot(tS, perspective[1:], label="perspective")

        axs[0].set_title("ankle velocity")
        axs[1].set_title("toe velocity")
        axs[2].set_title("heel velocity")
        axs[3].set_title("ground contact diff (gait events)")
        axs[0].set_yticks([0, 3, ankle_thr])
        axs[1].set_yticks([0, 3, big_toe_thr])
        axs[2].set_yticks([0, 3, heel_thr])
        axs[3].set_yticks([-1,1,],["FC", "IC"],)
        axs[3].set_yticks([-1,1,],["FC", "IC"],)
        axs[4].set_yticks([0, 1], ["back", "front"])

        for ax in axs:
            ax.legend(loc="upper left")
            ax.grid()

        plt.tight_layout()
        plt.savefig(filename)
        plt.close("all")
        print(f"figure saved to: \n{filename}")
        
        #fmt: on
    return (ICs, FCs, mean_velocity)


def get_gait_events(
    keypoint_data: np.ndarray,
    kpt_labels: list,
    gait_analysis_properties: dict,
    debug_figs_file_path: str = None,
) -> dict:
    """
    Compute all gait-events (IC,FC) for both sides (left, right) of a given recording.

    Args:
        keypoint_data: interpolated and filtered keypoint data (keypoints x dims x frames)
        kpt_labels: list of keypoint labels corresponding to data (left_ankle, nose, etc)
        gait_analysis_properties: dictionary of static parameters from gait_analysis_properties.json
        file_path: path to save debug figures

    Returns:
        gait_events: dict containing all IC and FC events (for both sides)

    """
    fps = gait_analysis_properties["fps"]

    # get turn mask (turn / straight segments) and perspectives (front/back)
    turn_mask, perspective = get_turns_and_perspective(
        keypoint_data=keypoint_data,
        kpt_labels=kpt_labels,
        sampling_fr=fps,
        min_peak_height=0.3,
        min_walk_duration=2,
        debug_fig_file_path=debug_figs_file_path,
    )

    events = {"left": {}, "right": {}}

    # get gait-events for both sides. Use 1 m/s as the initial velocity estimate
    for side in ["left", "right"]:
        _, _, velocity = get_gait_events_one_side(
            keypoint_data,
            kpt_labels,
            side,
            gait_analysis_properties,
            1,
            turn_mask,
            perspective,
        )
        ICs, FCs, _ = get_gait_events_one_side(
            keypoint_data,
            kpt_labels,
            side,
            gait_analysis_properties,
            velocity,
            turn_mask,
            perspective,
            debug_figs_file_path,
        )
        events[side]["ICs"] = ICs
        events[side]["FCs"] = FCs

    # combine perspective and turning mask to get segments for final readout
    segments = []
    for persp, turn in zip(perspective, turn_mask):
        match persp, turn:
            # front, straight
            case np.True_, np.False_:
                segment_name = "front_straight"
            # back, straight
            case np.False_, np.False_:
                segment_name = "back_straight"
            # all turns
            case _, _:
                segment_name = "turn"

        segments.append(segment_name)

    IC_events = [
        {"frame": frame, "side": side, "perspective": segments[frame]}
        for side in ["left", "right"]
        for frame in events[side]["ICs"]
    ]
    FC_events = [
        {"frame": frame, "side": side, "perspective": segments[frame]}
        for side in ["left", "right"]
        for frame in events[side]["FCs"]
    ]
    gait_events = {"IC": IC_events, "FC": FC_events}
    return gait_events

#-----------------------------------------GAIT ANALYSIS---------------------------------------------

def gait_analysis(
    keypoint_data: np.ndarray,
    gait_events: dict,
    kpt_labels: list,
    gait_analysis_properties: dict,
    debug_file_path: str = None,
) -> dict:

    # static properties for calculations
    fps = gait_analysis_properties["fps"]
    stride_min = gait_analysis_properties["stride_min"]
    stride_max = gait_analysis_properties["stride_max"]

    perspectives = ["all", "front_straight", "back_straight"]
    # metrics of interest (for each side)
    moi = {
        metric: []
        for metric in [
            "step_time",
            "step_length",
            "stride_time",
            "stride_length",
            "stride_velocity",
            "swing_time",
            "double_support_time",
            "base_of_support",
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

    # Initial- and Final-contact gait events
    ICs = gait_events["IC"]
    FCs = gait_events["FC"]

    num_steps_used = 0

    accepted_ICs = []
    rejected_no_full_cycle = []
    rejected_bad_stride_time = []
    rejected_bad_perspective = []
    rejected_missing_GEs = []

    GE_log = []
    GE_log.append(f"i, IC0, ipsi, perspective, IC0, FC0, IC1, FC1, IC2")

    # iterate Initial Contact gait events
    for i, IC in enumerate(ICs):

        # ipsilateral and contralateral sides for current gait event
        ipsi = IC["side"]
        contra = "left" if ipsi == "right" else "right"

        # frame idx of curernt IC
        IC0 = IC["frame"]

        # perspective value ('straight_front'/'straight_back') or 'all' if missing
        perspective = IC.get("perspective", "all")

        if perspective not in perspectives:
            rejected_bad_perspective.append(IC0)
            continue

        # get the next ipsilateral event's global index (relative to the whole event list), return None otherwise
        same_foot_next_idx = next(
            (
                j
                for j, event in enumerate(ICs[i + 1 :], start=i + 1)
                if event["side"] == ipsi and event["perspective"] == perspective
            ),
            None,
        )
        # check if there is a next ispi IC (full gait cycle for current foot)
        if same_foot_next_idx is not None:

            # stride time = time elapsed between heelstrikes of the same foot
            stride_time = (ICs[same_foot_next_idx]["frame"] - IC0) / fps

            # stride time falls in realistic time range
            if stride_min <= stride_time <= stride_max:
                # frame index of next IC event (ipsilateral)
                IC2 = ICs[same_foot_next_idx]["frame"]

                # frame index of first contralateral heel strike (between current and next ipsilateral)
                tolerance = fps * 0.2
                IC1 = get_frame_indices(ICs, contra, IC0, IC2, tolerance=tolerance)

                FC0, FC1 = None, None

                # if there's a next contralateral GE
                if IC1:
                    IC1 = IC1[0]

                    # see gait_events.png
                    FC0 = get_frame_indices(FCs, contra, IC0, IC1, tolerance=tolerance)
                    FC1 = get_frame_indices(FCs, ipsi, IC1, IC2, tolerance=tolerance)

                    FC0 = FC0[0] if FC0 else None
                    FC1 = FC1[0] if FC1 else None

                    # print(i, IC['frame'], ipsi, perspective, IC0, FC0, IC1, FC1, IC2)
                    GE_log.append(
                        f"{i}, {IC0}, {ipsi}, {perspective}, {IC0}, {FC0}, {IC1}, {FC1}, {IC2}"
                    )

                # if either of the values is None, skip cycle
                if any(x is None for x in [IC0, IC1, IC2, FC0, FC1]):
                    rejected_missing_GEs.append(IC0)
                    continue

                accepted_ICs.append(IC0)

                IC0_heel_position = keypoint_data[
                    kpt_labels.index(f"{ipsi}_heel"), :, IC0
                ]
                IC1_heel_position = keypoint_data[
                    kpt_labels.index(f"{contra}_heel"), :, IC1
                ]
                IC2_heel_position = keypoint_data[
                    kpt_labels.index(f"{ipsi}_heel"), :, IC2
                ]

                # step time [sec] = ipsi IC -> contra IC, IC1-IC0
                step_time = (IC1 - IC0) / fps

                # swing time [sec] = ipsi FC -> next ispi IC, IC2-FC1
                swing_time = (IC2 - FC1) / fps

                # step length [m] = ipsi IC-> contra IC, IC1-IC0
                step_length = np.linalg.norm(IC1_heel_position - IC0_heel_position)

                # stride lenght [m] = ipsi IC -> ipsi IC, IC2-IC0
                stride_length = np.linalg.norm(IC2_heel_position - IC0_heel_position)

                # stride velocity [m/s] = stride lenght / stride time
                stride_velocity = stride_length / stride_time

                # double support time [sec]= (ipsi IC-> contra FC) + (conrta IC->ispi FC), (FC0-IC0)+(FC1-IC1)
                double_support_time = ((FC0 - IC0) + (FC1 - IC1)) / fps

                # base of support [m] = see docs
                # norm(np.cross(p2-p1, p1-p3))/norm(p2-p1)
                # line connecting consecutive ispi heel positions
                IC2_IC0_line = IC2_heel_position - IC0_heel_position
                IC0_IC1_line = IC0_heel_position - IC1_heel_position

                base_of_support = np.linalg.norm(
                    np.cross(IC2_IC0_line, IC0_IC1_line)
                ) / np.linalg.norm(IC2_IC0_line)

                num_steps_used += 1
                for pers in ["all", perspective]:
                    metrics[pers][ipsi]["step_time"].append(step_time)
                    metrics[pers][ipsi]["step_length"].append(step_length)
                    metrics[pers][ipsi]["stride_time"].append(stride_time)
                    metrics[pers][ipsi]["stride_length"].append(stride_length)
                    metrics[pers][ipsi]["stride_velocity"].append(stride_velocity)
                    metrics[pers][ipsi]["swing_time"].append(swing_time)
                    metrics[pers][ipsi]["double_support_time"].append(
                        double_support_time
                    )
                    metrics[pers][ipsi]["base_of_support"].append(base_of_support)
            else:
                rejected_bad_stride_time.append(IC0)

        else:
            rejected_no_full_cycle.append(IC0)
    parameters = {}
    for perspective, data in metrics.items():
        parameters[perspective] = {
            metric: {
                **get_mean_and_cv(data["left"][metric], data["right"][metric]),
                "asymmetry": compute_asymmetry(
                    data["left"][metric], data["right"][metric]
                ),
            }
            for metric in data["left"]
        }

    if debug_file_path != None:

        debug_fig_file_path = f"{os.path.splitext(debug_file_path)[0]}_GE_analysis"
        debug_log_file_path = f"{os.path.splitext(debug_file_path)[0]}_log.txt"

        turn_mask, perspective = get_turns_and_perspective(
            keypoint_data=keypoint_data,
            kpt_labels=kpt_labels,
            sampling_fr=fps,
            min_peak_height=0.3,
            min_walk_duration=2,
        )
        accepted_ICs = np.array(accepted_ICs)
        rejected_no_full_cycle = np.array(rejected_no_full_cycle)
        rejected_bad_stride_time = np.array(rejected_bad_stride_time)
        rejected_bad_perspective = np.array(rejected_bad_perspective)
        rejected_missing_GEs = np.array(rejected_missing_GEs)

        t = np.linspace(0, keypoint_data.shape[2], keypoint_data.shape[2])

        accepted_vis = np.zeros_like(t)
        rejected_cycle_vis = np.zeros_like(t)
        rejected_stride_vis = np.zeros_like(t)
        rejected_perspective_vis = np.zeros_like(t)
        rejected_missing_vis = np.zeros_like(t)

        if rejected_no_full_cycle.shape != (0,):
            rejected_cycle_vis[rejected_no_full_cycle] = 1
        if rejected_bad_stride_time.shape != (0,):
            rejected_stride_vis[rejected_bad_stride_time] = 1
        if rejected_bad_perspective.shape != (0,):
            rejected_perspective_vis[rejected_bad_perspective] = 1
        if rejected_missing_GEs.shape != (0,):
            rejected_missing_vis[rejected_missing_GEs] = 1
        if accepted_ICs.shape != (0,):
            accepted_vis[accepted_ICs] = 1

        plt.close("all")
        fig, axs = plt.subplots(2, 1, figsize=(14, 3))
        axs[0].plot(t, rejected_missing_vis, "r", label="missing GEs")
        axs[0].plot(t, rejected_stride_vis, "tab:pink", label="bad stride time")
        axs[0].plot(t, rejected_cycle_vis, "tab:brown", label="no full cycle")
        if rejected_bad_perspective.shape != (0,):
            axs[0].plot(
                t, rejected_perspective_vis, "tab:gray", label="wrong perspective"
            )
        axs[0].plot(t, accepted_vis, "b")
        axs[0].plot(turn_mask, "k--", alpha=0.35, label="turn mask")
        axs[1].plot(t, perspective, label="perspective")

        axs[0].set_title("GEs used/rejected during analysis")
        axs[0].legend()
        axs[1].legend()
        axs[1].set_yticks([0, 1], ["back", "front"])

        plt.tight_layout()
        plt.show()
        plt.savefig(debug_fig_file_path)
        # plt.close("all")
        print(f"figure saved to: \n{debug_fig_file_path}")

        with open(debug_log_file_path,"w") as f:
            for line in GE_log:
                f.write(line+"\n")
    print(f"steps detected: {len(ICs)}")
    print(f"steps analyzed: {len(accepted_ICs)}")

    return parameters


def get_frame_indices(
    gait_events: list,
    side: Literal["left", "right"],
    lower_bound: int,
    upper_bound: int,
    tolerance: int,
) -> list:
    """
    Return frame indices from gait_events with the given side from lower_bound to upper_bound frame index.

    Args:
        gait_events: input data
        side: 'left' or 'right'
        lower_bound: lower frame index
        upper_bound: upper frame index
        tolerance: amount of time (in samples) with which to extend the search boundaries (both ways)
    Returns:
        list of gait events
    """
    return [
        event["frame"]
        for event in gait_events
        if event["side"] == side
        and lower_bound - tolerance <= event["frame"] <= upper_bound + tolerance
    ]


def get_mean_and_cv(left_values: np.ndarray, right_values: np.ndarray):
    """
    Calculates the mean and the coefficient of variation (cv [%]) for both sides (left,right) together.

    Args:
        left_values: values belonging to the left side
        right_values: values belonging to the right side

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


def display_results(parameters):
    parameter_order = [
        "step_time",
        "step_length",
        "stride_time",
        "stride_length",
        "stride_velocity",
        "swing_time",
        "double_support_time",
        "base_of_support",
    ]
    statistic_order = ["Mean [m]", "CV [%]", "Asymmetry [%]"]

    rows = []
    for segment_type, metrics in parameters.items():
        for param, values in metrics.items():
            rows.extend(
                [
                    {
                        "Parameter": param,
                        "Statistic": statistic_order[0],
                        "Perspective": segment_type,
                        "Value": values["mean"],
                    },
                    {
                        "Parameter": param,
                        "Statistic": statistic_order[1],
                        "Perspective": segment_type,
                        "Value": values["CV"],
                    },
                    {
                        "Parameter": param,
                        "Statistic": statistic_order[2],
                        "Perspective": segment_type,
                        "Value": values["asymmetry"],
                    },
                ]
            )
    pd.options.display.float_format = "{:,.2f}".format
    df = pd.DataFrame(rows)
    df["Parameter"] = pd.Categorical(
        df["Parameter"], categories=parameter_order, ordered=True
    )
    df["Statistic"] = pd.Categorical(
        df["Statistic"], categories=statistic_order, ordered=True
    )
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
    return (df, table)
