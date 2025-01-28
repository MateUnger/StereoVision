import sys
import pyzed.sl as sl
import numpy as np
import cv2
from pathlib import Path
import enum
import argparse
import os


def progress_bar(percent_done, bar_length=50):
    # Display a progress bar
    done_length = int(bar_length * percent_done / 100)
    bar = "=" * done_length + "-" * (bar_length - done_length)
    sys.stdout.write("[%s] %i%s\r" % (bar, percent_done, "%"))
    sys.stdout.flush()


def main():
    # Get input parameters
    svo_input_path = "./stereo_videos/straight_walk.svo2"
    output_dir = "./stereo_videos"
    avi_output_path = "./stereo_videos/straight_walk_nnnnn.avi"

    if not os.path.isdir(output_dir):
        sys.stdout.write(
            "Input directory doesn't exist. Check permissions or create it.\n",
            output_dir,
            "\n",
        )
        exit()

    # Specify SVO path parameter
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(svo_input_path)
    init_params.svo_real_time_mode = False  # Don't convert in realtime
    init_params.coordinate_units = sl.UNIT.CENTIMETER  # CENTIMETER, METER, MILLIMETER

    # Create ZED objects
    zed = sl.Camera()

    # Open the SVO file specified as a parameter
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        sys.stdout.write(repr(err))
        zed.close()
        exit()

    # Get image size
    image_size = zed.get_camera_information().camera_configuration.resolution
    width = image_size.width
    height = image_size.height

    # Prepare single image containers
    left_image = sl.Mat()
    # right_image = sl.Mat()
    depth_image = sl.Mat()

    video_writer = None

    # Create video writer with MPEG-4 part 2 codec
    video_writer = cv2.VideoWriter(
        avi_output_path,
        cv2.VideoWriter_fourcc(*"MJPG"),
        zed.get_camera_information().camera_configuration.fps,
        (width, height),
    )

    if not video_writer.isOpened():
        sys.stdout.write(
            "OpenCV video writer cannot be opened. Please check the .avi file path and write "
            "permissions.\n"
        )
        zed.close()
        exit()

    rt_param = sl.RuntimeParameters()

    # Start SVO conversion to AVI/SEQUENCE
    sys.stdout.write("Converting SVO... Use Ctrl-C to interrupt conversion.\n")

    nb_frames = zed.get_svo_number_of_frames()

    while True:
        err = zed.grab(rt_param)
        if err == sl.ERROR_CODE.SUCCESS:
            svo_position = zed.get_svo_position()

            # Retrieve SVO images
            zed.retrieve_image(left_image, sl.VIEW.LEFT)
            zed.retrieve_measure(depth_image, sl.MEASURE.DEPTH)

            # Convert SVO image from RGBA to RGB
            ocv_image_rgb = left_image.get_data()
            # print(ocv_image_rgb.shape)
            ocv_image_rgb = cv2.cvtColor(ocv_image_rgb, cv2.COLOR_BGRA2RGB)

            # Write the RGB image in the video
            video_writer.write(ocv_image_rgb)
            cv2.imshow("image", ocv_image_rgb)

            # Display progress
            progress_bar((svo_position + 1) / nb_frames * 100, 30)

        if err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            progress_bar(100, 30)
            sys.stdout.write("\nSVO end has been reached. Exiting now.\n")
            break

        video_writer.release()

    zed.close()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    main()
