import sys
import numpy as np
import pyzed.sl as sl
import cv2


def main():

# Create a Camera object
    zed = sl.Camera()

    #Set initial parameters
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE  # Use PERFORMANCE depth mode
    init_params.coordinate_units = sl.UNIT.METER  # Use meter units (for depth measurements)
    init_params.camera_resolution = sl.RESOLUTION.HD720 #Using 720 our resolution is 1260x720


    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    #Obtain the depth 
    depth_zed = sl.Mat(zed.get_camera_information().camera_resolution.width, zed.get_camera_information().camera_resolution.height, sl.MAT_TYPE.F32_C1)

    while zed.grab() == sl.ERROR_CODE.SUCCESS :
            # Retrieve depth data (32-bit)
            zed.retrieve_measure(depth_zed, sl.MEASURE.DEPTH)
            # Load depth data into a numpy array
            depth_ocv = depth_zed.get_data()
            # Print the depth matrix for entire image this is 1260x720 = 921600. This is a [1x921600] array
            print((depth_ocv))  

    # Close the camera
    zed.close()

if __name__ == "__main__":
    main()


