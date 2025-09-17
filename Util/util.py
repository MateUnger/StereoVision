import sys
import numpy as np
import warnings
import numpy as np
import importlib
from scipy.signal import butter, filtfilt
from scipy.interpolate import CubicSpline


rtmlib_module = importlib.import_module("rtmlib")


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


def compute_iou(bboxA, bboxB):
    """Compute the Intersection over Union (IoU) between two boxes .

    Args:
        bboxA (list): The first bbox info (left, top, right, bottom, score).
        bboxB (list): The second bbox info (left, top, right, bottom, score).

    Returns:
        float: The IoU value.
    """

    x1 = max(bboxA[0], bboxB[0])
    y1 = max(bboxA[1], bboxB[1])
    x2 = min(bboxA[2], bboxB[2])
    y2 = min(bboxA[3], bboxB[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    bboxA_area = (bboxA[2] - bboxA[0]) * (bboxA[3] - bboxA[1])
    bboxB_area = (bboxB[2] - bboxB[0]) * (bboxB[3] - bboxB[1])
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
    """Pose tracker for wholebody pose estimation.

    Args:
        solution (type): rtmlib solutions, e.g. Wholebody, Body, etc.
        det_frequency (int): Frequency of object detection.
        mode (str): 'performance', 'lightweight', or 'balanced'.
        to_openpose (bool): Whether to use openpose-style skeleton.
        backend (str): Backend of pose estimation model.
        device (str): Device of pose estimation model.
    """

    MIN_AREA = 1000

    def __init__(
        self,
        solution: type,
        det_frequency: int = 1,
        tracking: bool = True,
        tracking_thr: float = 0.3,
        mode: str = "performance",
        to_openpose: bool = False,
        backend: str = "onnxruntime",
        device: str = "gpu",
    ):

        model = solution(mode=mode, to_openpose=to_openpose, backend=backend, device=device)

        self.det_model = model.det_model
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

        if self.frame_cnt % self.det_frequency == 0:
            bboxes = self.det_model(image)
        else:
            bboxes = self.bboxes_last_frame

        keypoints, scores = self.pose_model(image, bboxes=bboxes)

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

        self.bboxes_last_frame = bboxes_current_frame
        self.frame_cnt += 1

        return keypoints, scores

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
    data: np.ndarray, sampling_rate: float, cutoff: float, order: int, gap_size: int
) -> np.ndarray:
    """
    Interpolates and applies a low-pass filter to data (with NaNs).

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
    filtered_data = data.copy()

    # Design Butterworth low-pass filter
    b, a = butter(N=order, Wn=cutoff / (0.5 * sampling_rate), btype="low", analog=False)

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
