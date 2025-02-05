import os 
import sys
import pyzed.sl as sl
from signal import signal, SIGINT


output_folder = './stereo_videos'
output_file_name = '50cm_walk_non_flicker_1_SVGA120FPS_low_pos.svo2'
cam = sl.Camera()

#Handler to deal with CTRL+C properly
def handler(signal_received, frame):
    cam.disable_recording()
    cam.close()
    sys.exit(0)

signal(SIGINT, handler)

def main():
    
    init = sl.InitParameters(
        camera_resolution = sl.RESOLUTION.SVGA, #HD1200, HD1080, SVGA
        camera_fps = 120, #60,30,15, 120(SVGA only)
        )
    
    init.async_image_retrieval = True; # This parameter can be used to record SVO in camera FPS even if the grab loop is running at a lower FPS (due to compute for ex.)

    status = cam.open(init) 
    if status != sl.ERROR_CODE.SUCCESS: 
        print("Camera Open", status, "Exit program.")
        exit(1)
        
    recording_param = sl.RecordingParameters(os.path.join(output_folder, output_file_name), sl.SVO_COMPRESSION_MODE.H264) # Enable recording with the filename specified in argument
    err = cam.enable_recording(recording_param)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Recording ZED : ", err)
        exit(1)

    runtime = sl.RuntimeParameters()
    print("SVO is Recording, use Ctrl-C to stop.") # Start recording SVO, stop with Ctrl-C command
    frames_recorded = 0

    while True:
        if cam.grab(runtime) == sl.ERROR_CODE.SUCCESS : # Check that a new image is successfully acquired
            frames_recorded += 1
            print("Frame count: " + str(frames_recorded), end="\r")
    
if __name__ == "__main__":
 
    main()