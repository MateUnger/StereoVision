"""
This sample demonstrates how to capture a live 3D point cloud
with the ZED SDK and display the result in an OpenGL window.
"""

import sys
import ogl_viewer.viewer as gl
import pyzed.sl as sl
import argparse


print("Running Depth Sensing sample ... Press 'Esc' to quit\nPress 's' to save the point cloud")

# Determine memory type based on CuPy availability and user preference
use_gpu = gl.GPU_ACCELERATION_AVAILABLE
mem_type = sl.MEM.GPU if use_gpu else sl.MEM.CPU
if use_gpu:
    print("🚀 Using GPU data transfer with CuPy")

zed = sl.Camera()
# mem_type = sl.MEM.GPU
input_type = sl.InputType()
input_type.set_from_svo_file(
    "stereo_videos\\validation_test\\43916681.svo2"
)  # Set init parameter to run from the .svo
init_parameters = sl.InitParameters(input_t=input_type, svo_real_time_mode=False)

init_parameters.depth_mode = sl.DEPTH_MODE.NEURAL_PLUS
init_parameters.coordinate_units = sl.UNIT.METER  # CENTIMETER, METER, MILLIMETER
init_parameters.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
init_parameters.depth_stabilization = 50  # 0 to trun it off, otherwise 1-100 linear. default is 30

# zed = sl.Camera()
status = zed.open(init_parameters)
if status != sl.ERROR_CODE.SUCCESS:
    print(repr(status))
    exit()

res = sl.Resolution()
res.width = -1
res.height = -1

# Get the first PC to retrieve the resolution
point_cloud = sl.Mat()
zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, mem_type, res)
res = point_cloud.get_resolution()

# Create OpenGL viewer
viewer = gl.GLViewer()
viewer.init(1, sys.argv, res)

while viewer.is_available():
    print("hey")
    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        # Retrieve point cloud data using the optimal memory type (GPU if CuPy available)
        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, mem_type, res)
        viewer.updateData(point_cloud)
        if viewer.save_data:
            # For saving, we take CPU memory regardless of processing type
            point_cloud_to_save = sl.Mat()
            zed.retrieve_measure(point_cloud_to_save, sl.MEASURE.XYZRGBA, mem_type)
            err = point_cloud_to_save.write("Pointcloud.ply")
            if err == sl.ERROR_CODE.SUCCESS:
                print("Current .ply file saving succeed")
            else:
                print("Current .ply file failed")
            viewer.save_data = False
viewer.exit()
zed.close()
