import sys
import numpy as np
import pyzed.sl as sl
import cv2
from matplotlib import pyplot as plt
import time
import os



def main():

########################################### Low Def Depth Image Capture ##################################################

# Create a Camera object
    zed = sl.Camera()

    #Set initial parameters
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE   # Use PERFORMANCE depth mode
    init_params.coordinate_units = sl.UNIT.METER  # Use meter units (for depth measurements)
    init_params.camera_resolution = sl.RESOLUTION.HD720 #Using 720 our resolution is 1260x720


    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    # Create a greyscale sl.Mat object
    image_depth_zed = sl.Mat(zed.get_camera_information().camera_resolution.width, zed.get_camera_information().camera_resolution.height, sl.MAT_TYPE.U8_C4)

    if zed.grab() == sl.ERROR_CODE.SUCCESS :

            # Retrieve the normalized depth image
            zed.retrieve_image(image_depth_zed, sl.VIEW.DEPTH)
            
            # Use get_data() to get the numpy array
            image_depth_ocv = image_depth_zed.get_data()
            
            #Get the greyscale image
            gray = cv2.cvtColor(image_depth_ocv, cv2.COLOR_BGR2GRAY)
            cv2.waitKey(0)
            
            # Saving the image
            
            # Find the number of images in the specified directory
            total_images = np.size(os.listdir('/home/shane/python_images/low_light/low_depth'))
            # Increase the image count by one
            add_image = total_images + 1
            # Append the image counter to the image name
            new_image_num = str(add_image)
            image_name = "image_"+new_image_num+'.png'
            # Form the final file path to save the new image to
            file_path = os.path.join('/home/shane/python_images/low_light/low_depth',image_name)

            # Write the grayscale image to the file path
            cv2.imwrite(file_path, gray)
            # Terminate cv2 windows
            cv2.destroyAllWindows()

    # Close the camera
    zed.close()

########################################### Left/Right Low Res Image Capture ##################################################

    # Create a ZED camera object
    zed = sl.Camera()

    # Set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720  


    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    # Grab an image
    runtime_parameters = sl.RuntimeParameters()

    

    # Grab an image, a RuntimeParameters object must be given to grab()
    if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        image_zed = sl.Mat(zed.get_camera_information().camera_resolution.width, zed.get_camera_information().camera_resolution.height, sl.MAT_TYPE.U8_C4)
    
        zed.retrieve_image(image_zed,sl.VIEW.LEFT)
        left = image_zed.get_data()
       
    # Find the number of images in the specified directory
        total_images = np.size(os.listdir('/home/shane/python_images/low_light/low_left'))
    # Increase the image count by one
        add_image = total_images + 1
    # Append the image counter to the image name
        new_image_num = str(add_image)
        image_name = "image_"+new_image_num+'.png'
    # Form the final file path to save the new image to
        file_path_left = os.path.join('/home/shane/python_images/low_light/low_left',image_name)

    # Write the grayscale image to the file path
        cv2.imwrite(file_path_left, left)
    # Terminate cv2 windows
        cv2.destroyAllWindows()

    # Saving the image
        zed.retrieve_image(image_zed,sl.VIEW.RIGHT)
        right = image_zed.get_data()
     
    # Find the number of images in the specified directory
        total_images = np.size(os.listdir('/home/shane/python_images/low_light/low_right'))
    # Increase the image count by one
        add_image = total_images + 1
        # Append the image counter to the image name
        new_image_num = str(add_image)
        image_name = "image_"+new_image_num+'.png'
    # Form the final file path to save the new image to
        file_path_right = os.path.join('/home/shane/python_images/low_light/low_right',image_name)

    # Write the grayscale image to the file path
        cv2.imwrite(file_path_right, right)
    # Terminate cv2 windows
        cv2.destroyAllWindows()
    
    
     

        input_anything = input('Press any key to continue to high quality image capture') 
        print('Continuing on to execute high quality image capture')


########################################### High Def Depth Image Capture ##################################################
# Create a Camera object
    zed = sl.Camera()

    #Set initial parameters
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL  # Use PERFORMANCE depth mode
    init_params.coordinate_units = sl.UNIT.METER  # Use meter units (for depth measurements)
    init_params.camera_resolution = sl.RESOLUTION.HD720 #Using 720 our resolution is 1260x720


    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    # Create a greyscale sl.Mat object
    image_depth_zed = sl.Mat(zed.get_camera_information().camera_resolution.width, zed.get_camera_information().camera_resolution.height, sl.MAT_TYPE.U8_C4)

    if zed.grab() == sl.ERROR_CODE.SUCCESS :

            # Retrieve the normalized depth image
            zed.retrieve_image(image_depth_zed, sl.VIEW.DEPTH)
            
            # Use get_data() to get the numpy array
            image_depth_ocv = image_depth_zed.get_data()
            
            #Get the greyscale image
            gray = cv2.cvtColor(image_depth_ocv, cv2.COLOR_BGR2GRAY)
            cv2.waitKey(0)
            
            # Saving the image
            
            # Find the number of images in the specified directory
            total_images = np.size(os.listdir('/home/shane/python_images/high_light/high_depth'))
            # Increase the image count by one
            add_image = total_images + 1
            # Append the image counter to the image name
            new_image_num = str(add_image)
            image_name = "image_"+new_image_num+'.png'
            # Form the final file path to save the new image to
            file_path = os.path.join('/home/shane/python_images/high_light/high_depth',image_name)

            # Write the grayscale image to the file path
            cv2.imwrite(file_path, gray)
            # Terminate cv2 windows
            cv2.destroyAllWindows()
           
########################################### Left/Right High Res Image Capture ##################################################

    # Create a ZED camera object
    zed = sl.Camera()

    # Set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720  


    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    # Grab an image
    runtime_parameters = sl.RuntimeParameters()

    

    # Grab an image, a RuntimeParameters object must be given to grab()
    if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        image_zed = sl.Mat(zed.get_camera_information().camera_resolution.width, zed.get_camera_information().camera_resolution.height, sl.MAT_TYPE.U8_C4)
    
        zed.retrieve_image(image_zed,sl.VIEW.LEFT)
        left = image_zed.get_data()
    
       
    # Find the number of images in the specified directory
        total_images = np.size(os.listdir('/home/shane/python_images/high_light/high_left'))
    # Increase the image count by one
        add_image = total_images + 1
    # Append the image counter to the image name
        new_image_num = str(add_image)
        image_name = "image_"+new_image_num+'.png'
    # Form the final file path to save the new image to
        file_path_left = os.path.join('/home/shane/python_images/high_light/high_left',image_name)

    # Write the grayscale image to the file path
        cv2.imwrite(file_path_left, left)
    # Terminate cv2 windows
        cv2.destroyAllWindows()

    # Saving the image
        zed.retrieve_image(image_zed,sl.VIEW.RIGHT)
        right = image_zed.get_data()
  
    # Find the number of images in the specified directory
        total_images = np.size(os.listdir('/home/shane/python_images/high_light/high_right'))
    # Increase the image count by one
        add_image = total_images + 1
        # Append the image counter to the image name
        new_image_num = str(add_image)
        image_name = "image_"+new_image_num+'.png'
    # Form the final file path to save the new image to
        file_path_right = os.path.join('/home/shane/python_images/high_light/high_right',image_name)

    # Write the grayscale image to the file path
        cv2.imwrite(file_path_right, right)
    # Terminate cv2 windows
        cv2.destroyAllWindows()

    # Close the camera
    zed.close()

    



if __name__ == "__main__":
    main()


