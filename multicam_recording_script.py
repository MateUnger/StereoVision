import os
import cv2
import time
import signal
import threading
import numpy as np
import pyzed.sl as sl
import matplotlib.pyplot as plt

zed_list = []
current_images = []
current_timestamps = []
thread_list = []
stop_signal = False

def signal_handler(signal, frame):
    global stop_signal
    stop_signal=True
    time.sleep(0.5)
    exit()

def grab_run(index):
    global stop_signal
    global zed_list
    global current_timestamps
    global current_images
    global name_list

    runtime = sl.RuntimeParameters()
    while not stop_signal:

        err = zed_list[index].grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS:

            zed_list[index].retrieve_image(current_images[index], sl.VIEW.LEFT)
            current_timestamps[index] = zed_list[index].get_timestamp(sl.TIME_REFERENCE.CURRENT).data_ns #time of function call
        # barrier.wait()

        time.sleep(0.0001) #1ms
    zed_list[index].disable_recording()
    zed_list[index].close()

def main():
    global stop_signal
    global zed_list
    global current_images
    global current_timestamps
    global thread_list
    global name_list
    global barrier
    signal.signal(signal.SIGINT, signal_handler)

    output_file_names = ['multicam_1.svo2','multicam_2.svo2']

    print("Running...")
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.SVGA
    init.camera_fps = 120  # The framerate is lowered to avoid any USB3 bandwidth issues
    init.async_image_retrieval = False

    #List and open cameras
    name_list = []
    last_ts_list = []
    cameras = sl.Camera.get_device_list()

    index = 0
    for cam in cameras:
        init.set_from_serial_number(cam.serial_number) # define input source
        name_list.append(f"ZED_{cam.serial_number}")

        print(f"Opening {name_list[index]}")
        zed_list.append(sl.Camera())
        current_images.append(sl.Mat())
        current_timestamps.append(0)
        last_ts_list.append(0)

        # open cameras
        status = zed_list[index].open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            print(repr(status))
            zed_list[index].close()

        # set recording params
        recording_param = sl.RecordingParameters(
            os.path.join('./', output_file_names[index]),
            sl.SVO_COMPRESSION_MODE.H264) # Enable recording with the filename specified in argument
        err = zed_list[index].enable_recording(recording_param)
        index = index +1

        if err != sl.ERROR_CODE.SUCCESS:
            print("Recording ZED : ", err)
            exit(1)

    num_threads = len(zed_list)
    barrier = threading.Barrier(num_threads)

    #Start camera threads
    for index in range(0, len(zed_list)):
        if zed_list[index].is_opened():
            thread_list.append(threading.Thread(target=grab_run, args=(index,)))
            thread_list[index].start()
    
    #Display camera images
    key = ''
    while key != 113:  # for 'q' key
        for cam_index in range(0, len(zed_list)):
            if zed_list[cam_index].is_opened():

                # if there is a new frame 
                if (current_timestamps[cam_index] >= last_ts_list[cam_index]):
                    img = current_images[cam_index].get_data()

                    img = cv2.circle(img,(960,540),10,(0,0,255),3)
                    
                    cv2.imshow(name_list[cam_index], cv2.resize(img,(600,400)))

                    # set the last timestamp for each cam to the CURRENT_ts one of each cam 
                    last_ts_list[cam_index] = current_timestamps[cam_index]

        key = cv2.waitKey(10)
    cv2.destroyAllWindows()

    #Stop the threads
    stop_signal = True
    for index in range(0, len(thread_list)):
        thread_list[index].join()
    print("\nFINISH")

if __name__ == "__main__":
    main()