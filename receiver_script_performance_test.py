"""
    Read a stream and display the left images using OpenCV
"""

import sys
import pyzed.sl as sl
import cv2
import argparse
import socket


def valid_ip_or_hostname(ip_or_hostname):
    try:
        host, port = ip_or_hostname.split(":")
        socket.inet_aton(host)  # Vérifier si c'est une adresse IP valide
        port = int(port)
        return f"{host}:{port}"
    except (socket.error, ValueError):
        raise argparse.ArgumentTypeError(
            "Invalid IP address or hostname format. Use format a.b.c.d:p or hostname:p"
        )


ip_address = valid_ip_or_hostname("192.162.0.100:30000")

init_parameters = sl.InitParameters()
init_parameters.depth_mode = sl.DEPTH_MODE.NONE
init_parameters.sdk_verbose = 1
init_parameters.set_from_stream(ip_address.split(":")[0], int(ip_address.split(":")[1]))

cam = sl.Camera()
status = cam.open(init_parameters)

if status != sl.ERROR_CODE.SUCCESS:
    print("Camera Open : " + repr(status) + ". Exit program.")
    exit()

runtime = sl.RuntimeParameters()

print(f"camera fps: {init_parameters.camera_fps}")

win_name = "Camera Remote Control"
mat = sl.Mat()
cv2.namedWindow(win_name)

key = ""
while key != 113:  # for 'q' key
    err = cam.grab(runtime)  # Check that a new image is successfully acquired
    if err == sl.ERROR_CODE.SUCCESS:
        cam.retrieve_image(mat, sl.VIEW.LEFT)  # Retrieve left image
        fps = cam.get_current_fps()
        print(fps)
        cvImage = mat.get_data()
        cv2.imshow(win_name, cvImage)
    else:
        print("Error during capture : ", err)
        break
    key = cv2.waitKey(5)
cv2.destroyAllWindows()

cam.close()


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--ip_address', type=valid_ip_or_hostname, help='IP address or hostname of the sender. Should be in format a.b.c.d:p or hostname:p', required=True)
#     opt = parser.parse_args()
#     main()
