"""
Read .svo/.svo2 files and display the video(s)
"""

import os
import cv2
import pyzed.sl as sl
from Util.util import progress_bar


def main():

    input_folder = "./stereo_videos"
    input_file_name = "43916681.svo2"
    filepath = os.path.join(input_folder, input_file_name)

    input_type = sl.InputType()
    input_type.set_from_svo_file(filepath)  # Set init parameter to run from the .svo
    init = sl.InitParameters(input_t=input_type, svo_real_time_mode=False)

    cam = sl.Camera()
    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:  # Ensure the camera opened succesfully
        print("Camera Open", status, "Exit program.")
        exit(1)

    # Set a maximum resolution, for visualisation confort
    resolution = cam.get_camera_information().camera_configuration.resolution
    low_resolution = sl.Resolution(
        min(720, resolution.width) * 2, min(404, resolution.height)
    )
    svo_image = sl.Mat(
        min(720, resolution.width) * 2,
        min(404, resolution.height),
        sl.MAT_TYPE.U8_C4,
        sl.MEM.CPU,
    )

    runtime = sl.RuntimeParameters()

    mat = sl.Mat()

    svo_frame_rate = cam.get_init_parameters().camera_fps
    nb_frames = cam.get_svo_number_of_frames()

    print("[Info] SVO contains ", nb_frames, " frames")

    while True:
        err = cam.grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS:
            # cam.retrieve_image(
            #     svo_image, sl.VIEW.SIDE_BY_SIDE, sl.MEM.CPU, low_resolution
            # )
            cam.retrieve_measure(mat, sl.MEASURE.DEPTH)
            svo_position = cam.get_svo_position()
            cv2.imshow("View", svo_image.get_data())  # dislay both images to cv2

            if cv2.waitKey(25) & 0xFF == ord("q"):
                break

            progress_bar(svo_position / nb_frames * 100, 30)

        elif err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:  # Check if the .svo has ended
            progress_bar(100, 30)
            print("SVO end has been reached. Looping back to 0")
            cam.set_svo_position(0)
        else:
            print("Grab ZED : ", err)
            break
    cv2.destroyAllWindows()
    cam.close()


if __name__ == "__main__":

    main()
