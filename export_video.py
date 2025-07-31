"""
Convert .svo2 video into .avi video using the stream from the left camera.
"""

import sys
import pyzed.sl as sl
import cv2
import os
from .Util.util import progress_bar
from .Util.log_util import get_logger


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
    log = get_logger(os.path.basename(__file__).split(".")[0])

    # if there's no output path specified, use the input path
    if output_path == "default":
        output_path = input_path.split(".")[0] + ".avi"

    if not os.path.exists(input_path):
        log.error(f"Input directory doesn't exist:b{input_path}")
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
        log.error(repr(err))
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
        log.error("OpenCV video writer cannot be opened")
        zed.close()
        sys.exit()

    rt_param = sl.RuntimeParameters()

    # Start SVO conversion to AVI/SEQUENCE
    log.info(f"Converting {output_path}... num frames: {nb_frames}, framerate: {svo_frame_rate}")

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
            log.info(f"Done {output_path}.")
            break
    # Close the video writer
    video_writer.release()

    zed.close()
    return 0
