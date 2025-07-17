"""
Convert .svo2 video into .avi video using the stream from the left camera.
"""

import sys
import pyzed.sl as sl
import cv2
import os
from Util.util import progress_bar
from Util.log_util import get_logger


def main():

    # Initialize logger
    log = get_logger(os.path.basename(__file__).split(".")[0])

    # Get input parameters
    # input_folder = "c:/Users/unger/Work/Motion sensing/Code/alphapose/sl_vids/stick_8A1657ED-B002-4412-B6EA-04C8B2130D54"
    input_folder = "./stereo_videos/ken"
    # input_file_name = "43804892.svo2"

    # input_file_names = ["43804892.svo2", "43916681.svo2"]
    input_file_names = ["43804892.svo2"]

    output_folder = input_folder

    if not os.path.isdir(input_folder):
        log.error(f"Input directory doesn't exist:b{input_folder}")
        exit()

    # Check if output folder exists, if not create it
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    for file in input_file_names:
        # Specify SVO path parameter
        init_params = sl.InitParameters()
        init_params.set_from_svo_file(os.path.join(input_folder, file))
        init_params.svo_real_time_mode = False  # Don't convert in realtime
        output_file_name = file.split(".")[0] + ".avi"

        # Create ZED objects
        zed = sl.Camera()

        # Open the SVO file specified as a parameter
        err = zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            log.error(repr(err))
            zed.close()
            exit()

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
            os.path.join(output_folder, output_file_name),
            cv2.VideoWriter_fourcc(*"MPEG"),
            #    cv2.VideoWriter_fourcc(*'RGBA'),
            svo_frame_rate,
            (width, height),
        )
        if not video_writer.isOpened():
            log.error("OpenCV video writer cannot be opened")
            zed.close()
            exit()

        rt_param = sl.RuntimeParameters()

        # Start SVO conversion to AVI/SEQUENCE
        # sys.stdout.write("Converting SVO... Use Ctrl-C to interrupt conversion")
        log.info(
            f"Converting {os.path.join(input_folder, file)}... num frames: {nb_frames}, framerate: {svo_frame_rate}"
        )

        # print(f"total number of frames: {nb_frames}")
        # print(f"framerate of video:     {svo_frame_rate}")

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
                # sys.stdout.write("\n SVO end has been reached. Exiting now. \n")
                log.info(f"Done {os.path.join(input_folder, file)}.")
                break
        # Close the video writer
        video_writer.release()

        zed.close()
    return 0


if __name__ == "__main__":

    main()
