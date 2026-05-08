import os
import sys
import cv2
import numpy as np

import csv
import json
import pandas as pd

from scipy.signal import butter, filtfilt
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft
from scipy import signal as sp_signal

import pyzed.sl as sl
import copy
from typing import Literal

import importlib

rtmlib_module = importlib.import_module("rtmlib")


# -----------------------------------------QUALISYS PREPROCESSING---------------------------------------------


def format_qualisys_export(
    input_filename: str, output_filename: str = None
) -> tuple[np.ndarray, list]:
    # TODO: add logging, maybe return value
    """
    Read qualisys export file in .tsv format, format contents to fit multidimensional np.array,
    delete gap-filled entries and convert measurements from milimeter to meter.

    Args:
        input_filename: path to input file (.tsv)
        output_filename: optional, path to save the result as a .npz file. Default is the input_filename directory.

    Returns:
        keypoints: np.array of keypoint data
        labels: keypoint labels as list
    """
    if not os.path.exists(input_filename):
        raise FileNotFoundError(f"File {input_filename} does not exits")

    elif output_filename == None:
        basename = get_basename(input_filename)
        directory = os.path.dirname(input_filename)
        output_filename = os.path.join(directory, f"qualisys.npz")

    # read first 10 rows to get number of markers, frames, marker names...
    metadata = get_qualisys_metadata(input_filename)
    labels = metadata["marker_names"]

    # Read the TSV file, skipping the first 10 rows (only metadata)
    df = pd.read_csv(input_filename, sep="\t", header=None, skiprows=11, dtype=None)

    # Convert the DataFrame to a numpy array, transpose it (so it is (markers x dims) x frames)
    keypoints = df.to_numpy()[1:, :92].T

    # reshape the data to    markers x dims x frames
    keypoints = keypoints.reshape(int(metadata["NO_OF_MARKERS"]), 4, int(metadata["NO_OF_FRAMES"]))

    # iterate over all keypoints
    for kpt_idx, keypoint in enumerate(keypoints):
        # get the type of measurement for all frames of a given keypoint
        measurement_types = keypoint[3, :]
        # iterate over all dims (x,y,z,type) of a keypoint
        for dim_idx, dim in enumerate(keypoint):

            # replace coordinates with nan when it is gap-filled by qualisys, convert coords to float
            if dim_idx <= 2:
                keypoints[kpt_idx, dim_idx, :] = np.where(
                    measurement_types == "Measured", dim, "Nan"
                ).astype(float)

            # leave type as str ('Measured' or 'Gap-filled')
            elif dim_idx == 3:
                pass
    # remove dim 3 (type of measurement)
    keypoints = keypoints[:, :3, :].astype(float)

    # convert from milimeters to meters
    keypoints = keypoints / 1000

    # calculate mid_hip_front keypoint
    left_hip_front = keypoints[labels.index("left_hip_front"), :, :]
    right_hip_front = keypoints[labels.index("right_hip_front"), :, :]
    mid_hip_front = np.mean((left_hip_front, right_hip_front), axis=0)

    # calculate mid_hip_back keypoint
    left_hip_back = keypoints[labels.index("left_hip_back"), :, :]
    right_hip_back = keypoints[labels.index("right_hip_back"), :, :]
    mid_hip_back = np.mean((left_hip_back, right_hip_back), axis=0)

    # calculate mid_shoulder front keypoint
    left_shoulder = keypoints[labels.index("left_shoulder"), :]
    right_shoulder = keypoints[labels.index("right_shoulder"), :]
    mid_shoulder = np.mean((left_shoulder, right_shoulder), axis=0)

    # add new keypoints and labels to existing keypoints and labels
    keypoints = np.concatenate((keypoints, mid_shoulder[np.newaxis, :, :]), axis=0)
    keypoints = np.concatenate((keypoints, mid_hip_back[np.newaxis, :, :]), axis=0)
    keypoints = np.concatenate((keypoints, mid_hip_front[np.newaxis, :, :]), axis=0)
    labels.append("mid_shoulder")
    labels.append("mid_hip_back")
    labels.append("mid_hip_front")

    # np.savez(output_filename, keypoints=keypoints, labels=labels)
    return keypoints, labels


def get_qualisys_metadata(filename: str) -> dict:
    """
    Read qualisys export file in .tsv format, parse metadata.
    Exported file has to include tsv-header (qualisys export setting).

    Args:
        filename: path to the qualisys export file in .tsv format

    Returns:
        qualisys_metadata: metadata in dict format
    """

    qualisys_metadata = {}
    with open(filename) as f:
        # read file with csv reader. no need for pandas for just a few lines
        reader = csv.reader(f, delimiter="\t", quotechar='"')

        for ind, row in enumerate(reader):
            # only the first 9 rows of the whole file contain the tsv header metadata
            if ind <= 8:
                if ind < 6:
                    qualisys_metadata[row[0]] = float(row[1])
                # this row has 2 pieces of info; timestamp of the recording from qualisys (this cant be used for sync), timestamp from the start of host system
                elif ind == 7:
                    qualisys_metadata[row[0] + "_QUALISYS"] = row[1]
                    qualisys_metadata[row[0] + "_FROM_SYSTEM_START"] = row[2]
                else:
                    qualisys_metadata[row[0]] = row[1]
            if ind == 9:
                qualisys_metadata["marker_names"] = row[1:]

    return qualisys_metadata


# -----------------------------------------INTERPOLATION & FILTERING---------------------------------------------
def load_pose_model():
    """
    Load and configure 2D pose estimation model with preset settings.
    """

    # load model info (config, input size, etc)
    with open("./Util/2d_model/models.json") as f:
        models = json.load(f)

    device = "cuda"  # cpu, cuda, mps
    backend = "onnxruntime"  # opencv, onnxruntime, openvino
    detector_name = "YOLOX_nano"  # 'YOLOX_l_COCO','YOLOX_nano','YOLOX_tiny','YOLOX_s','YOLOX_m','YOLOX_l','YOLOX_x'
    pose_name = "RTMPose_x"  # (26) 'RTMPose_t', 'RTMPose_s', 'RTMPose_m', 'RTMPose_l', 'RTMPose_m2', 'RTMPose_l2', 'RTMPose_x', (133) 'RTMW_l', 'RTMW_x'
    labels = models["pose_models"]["26"]["kpt_labels"]

    custom_model = Custom(
        det_class="YOLOX",  #'RTMDet',
        det=models["detectors"][detector_name]["path"],
        det_input_size=models["detectors"][detector_name]["input_size"],
        pose_class="RTMPose",
        pose=models["pose_models"]["26"]["models"][pose_name]["path"],
        pose_input_size=models["pose_models"]["26"]["models"][pose_name]["input_size"],
        backend=backend,
        device=device,
    )

    return custom_model


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
                    np.flatnonzero(nans),
                    np.flatnonzero(valid_indices),
                    trajectory[valid_indices],
                )

            trajectory = filtfilt(b, a, trajectory)  # Apply filter
            trajectory[nans] = np.nan  # Restore NaNs

            filtered_data[kpt, dim, :] = trajectory

    return filtered_data


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
                np.flatnonzero(nan_indices),
                np.flatnonzero(valid_indices),
                coords[valid_indices],
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


# -----------------------------------------------------OTHER-----------------------------------------------------


def get_basename(full_file_path: str) -> str:
    """
    Return the basename of input file path e.g.: ...folder1/folder2/image_name.png -> image_name

    Args:
        full_file_path: input file path string

    Returns:
        basename: basename of the input (without folders or extension)
    """
    basename_without_ext = os.path.splitext(full_file_path)[0]
    basename = os.path.basename(basename_without_ext)
    return basename


def get_serial_number(filename: str) -> str:
    """
    get the camera serial number from filename

    Parameters:
        filename: should look like SN123456_...
    """

    return filename.split("_")[-1].split(".")[0]


def progress_bar(percent_done, bar_length=50):
    # Display a progress bar
    done_length = int(bar_length * percent_done / 100)
    bar = "=" * done_length + "-" * (bar_length - done_length)
    sys.stdout.write("[%s] %i%s\r" % (bar, percent_done, "%"))
    sys.stdout.flush()


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

    def __call__(self, image: np.ndarray):
        if self.one_stage:
            keypoints, scores = self.pose_model(image)
        else:
            bboxes = self.det_model(image)
            keypoints, scores = self.pose_model(image, bboxes=bboxes)

        return keypoints, scores


def export_svo_avi(input_path: str, output_path: str = "default") -> int:
    """ "
    Export .svo / .svo2 video to .avi format. Uses data from the left camera stream.

    Args:
        input_path (str): path of the input .svo2 file.
        output_path (str): path of the output .avi file. By default, it is the same as input path (with .avi extension)

    Returns:
        success: 1 if successful, 0 otherwise.

    """

    # Initialize logger
    # log = get_logger(os.path.basename(__file__).split(".")[0])

    # if there's no output path specified, use the input path
    if output_path == "default":
        output_path = input_path.split(".")[0] + ".avi"

    if not os.path.exists(input_path):
        # TODO: fix logging
        # log.error(f"Input directory doesn't exist:b{input_path}")
        sys.exit()

    # Check if output folder exists, if not create it
    # if not os.path.exists(output_path):
    #     os.makedirs(output_folder)

    # Specify SVO path parameter
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(input_path)
    init_params.svo_real_time_mode = False  # Don't convert in realtime

    # Create ZED objects
    zed = sl.Camera()

    # Open the SVO file specified as a parameter
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        # log.error(repr(err))
        zed.close()
        sys.exit()

    # Get image size
    image_size = zed.get_camera_information().camera_configuration.resolution
    width = image_size.width
    height = image_size.height

    zed.set_svo_position(0)
    nb_frames = zed.get_svo_number_of_frames()
    svo_frame_rate = zed.get_init_parameters().camera_fps

    # Prepare single image containers
    left_image = sl.Mat()

    # Create video writer with MPEG-4 part 2 codec
    video_writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"MPEG"),
        #    cv2.VideoWriter_fourcc(*'RGBA'),
        svo_frame_rate,
        (width, height),
    )
    if not video_writer.isOpened():
        # log.error("OpenCV video writer cannot be opened")
        zed.close()
        sys.exit()

    rt_param = sl.RuntimeParameters()

    # Start SVO conversion to AVI/SEQUENCE
    # log.info(f"Converting {output_path}... num frames: {nb_frames}, framerate: {svo_frame_rate}")

    while True:
        err = zed.grab(rt_param)
        if err == sl.ERROR_CODE.SUCCESS:
            svo_position = zed.get_svo_position()

            # Retrieve SVO images
            zed.retrieve_image(left_image, sl.VIEW.LEFT)

            svo_image_rgba = left_image.get_data()
            cv_image_rgb = cv2.cvtColor(svo_image_rgba, cv2.COLOR_RGBA2RGB)

            # cv2.imshow("View ", cv_image_rgb)
            video_writer.write(cv_image_rgb)

            # Display progress
            progress_bar((svo_position + 1) / nb_frames * 100, 30)
        if err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            progress_bar(100, 30)
            # log.info(f"Done {output_path}.")
            break
    # Close the video writer
    video_writer.release()

    zed.close()
    return 0


def normalize_vector(vector: np.ndarray):
    """
    Returns the unit vector of the input vector.

    Args:
        vector: input vector

    Returns:
        normalized_vector: unit vector of magnitude 1 with the same direction as the input
    """
    return vector / np.linalg.norm(vector)


def angle_between_vectors(vector_1: np.ndarray, vector_2: np.ndarray) -> float:
    """
    Returns the angle in degrees between vectors 'vector_1' and 'vector_2'

    Args:
        vector_1: input vector
        vector_2: other input vector
    Returns:
        angle: angle between the two input vectors in degrees.

    """
    # calculate unit vectors
    vector_1_unit = normalize_vector(vector_1)
    vector_2_unit = normalize_vector(vector_2)

    # get scalar product
    scalar_product = np.clip(np.dot(vector_1_unit, vector_2_unit), -1, 1)

    # clip scalar product to [-1,1] range so trig. function works normal,
    # use arccos to get the angle from scalar product
    angle = np.degrees(np.arccos(scalar_product))

    return angle


def project_vector_on_plane(plane_normal_vector: np.ndarray, vector: np.ndarray):
    """
    Project an n-dimensional vector onto an n-dimensional plane defined by its normal (orthogonal) vector.

    Args:
        plane_normal_vector: vector orthogonal to the reference plane
        vector: vector to be projected
    Returns:
        vector_projection: projection of the input vector onto reference plane
    """
    # normalize surface normal
    normal_vector = plane_normal_vector / np.linalg.norm(plane_normal_vector)
    # multily vector with its scalar product (dot product w surface normal), shift it
    vector_projection = vector - np.dot(vector, normal_vector) * normal_vector
    return vector_projection


def get_projection(plane_normal_vectors: np.ndarray, vector_array: np.ndarray):
    """
    Project each member of an array of vectors onto plane.

    Args:
        plane_normal_vector: vector orthogonal to the reference plane
        vector_array: array of vectros to be projected. Usually a limb for 1 gait-cycle

    Returns:
        array of projected vectors
    """

    return np.array(
        [
            project_vector_on_plane(plane_normal, vector)
            for plane_normal, vector in zip(plane_normal_vectors.T, vector_array.T)
        ]
    )


def load_keypoints_and_labels(npz_data_path: str) -> tuple[np.ndarray, list]:
    """
    Load keypoint and label data from .npz file
    File has to have keys: "keypoints", "labels"

    Args:
        npz_data_path: path to the .npz file

    Returns:
        keypoints: loaded keypoint data as np.array
        labels: loaded labels as list
    """
    if os.path.exists(npz_data_path):
        loaded_data = np.load(npz_data_path)
        keypoints = loaded_data["keypoints"]
        labels = loaded_data["labels"].tolist()
        print(f"loaded data from: {npz_data_path} with keys: {[key for key in loaded_data.keys()]}")
    else:
        raise FileNotFoundError(f"File: {npz_data_path} does not exist!")
    return keypoints, labels
