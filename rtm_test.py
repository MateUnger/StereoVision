import cv2
import numpy as np
from rtmlib import Wholebody, draw_skeleton
from StereoVision.Util.util import BodyWithFeet, PoseTracker, Body


device = "cuda"  # cpu, cuda, mps
backend = "onnxruntime"  # opencv, onnxruntime, openvino
openpose_skeleton = False  # True for openpose-style, False for mmpose-style


# estimation models
wholebody = Wholebody(
    to_openpose=openpose_skeleton,
    mode="balanced",  # 'performance', 'lightweight', 'balanced'. Default: 'balanced'
    backend=backend,
    device=device,
)

halpe26 = BodyWithFeet(to_openpose=openpose_skeleton, backend=backend, device=device)

pose_tracker = PoseTracker(
    Wholebody,
    det_frequency=1,  # detect every x frames
    to_openpose=openpose_skeleton,
    backend=backend,
    device=device,
)

body = Body(to_openpose=openpose_skeleton, backend=backend, device=device)

keypoints_over_time = []


cap = cv2.VideoCapture("./straight_walk.mp4")

if cap.isOpened() == False:
    print("Error opening video file")

# Read until video is completed
while cap.isOpened():

    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:

        # keypoints, scores = wholebody(frame)
        # keypoints, scores = halpe26(frame)
        # keypoints, scores = pose_tracker(frame)
        keypoints, scores = body(frame)
        keypoints_over_time.append(keypoints[0])

        # # if you want to use black background instead of original image,
        # img_show = np.zeros(frame.shape, dtype=np.uint8)
        img_show = draw_skeleton(frame, keypoints, scores, kpt_thr=0.5)

        cv2.imshow("Image", img_show)

        # Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

    else:
        break


cap.release()
cv2.destroyAllWindows()
