import sys
import numpy as np
import tifffile
import pyzed.sl as sl
import cv2
from matplotlib import pyplot as plt
import time
import os
from PIL import Image
from numpy import inf
import imageio
np.set_printoptions(threshold=sys.maxsize)


def main():

########################################### Low Def Depth Image Capture ##################################################

# Create a Camera object
    zed = sl.Camera()
    
    #Set initial parameters
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA   # Use ULTRA depth mode
    init_params.coordinate_units = sl.UNIT.METER  # Use meter units (for depth measurements)
    init_params.camera_resolution = sl.RESOLUTION.HD720 #Using 720 our resolution is 1260x720
    #print(zed)
    
    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
     exit(1)

    # Create a sl.Mat object of type 32 bit float
    image_zed = sl.Mat(zed.get_camera_information().camera_resolution.width, zed.get_camera_information().camera_resolution.height, sl.MAT_TYPE.F32_C1)
    if zed.grab() == sl.ERROR_CODE.SUCCESS :
            
             # Retrieve the depth image
            zed.retrieve_measure(image_zed, sl.MEASURE.DEPTH)
            
            # Use get_data() to get the numpy array both for manipulation and for image extraction
            depth = image_zed.get_data()
            depth = cv2.resize(depth, (640,360), cv2.INTER_LANCZOS4)
        

            ########### Saving the image ##############
            
            # Find the number of images in the specified directory
            total_images = np.size(os.listdir('/home/shane/final_python_images/low_light/low_depth_tif'))
            
            # Increase the image count by one
            add_image = total_images + 1
            
            # Append the image counter to the image name
            new_image_num = str(add_image)
            image_name_1 = "image_"+new_image_num+'.tif'
            image_name_2 = "image_"+new_image_num+'.png'
            
            # Form the final file path to save the new image to
            file_path_1 = os.path.join('/home/shane/final_python_images/low_light/low_depth_tif',image_name_1)
            file_path_2 = os.path.join('/home/shane/final_python_images/low_light/low_depth_png',image_name_2)
            maxd = 20
            mind = 0.5
            depthinterm = np.asarray(depth, dtype=np.uint8)
            depth2 = ((depthinterm - mind)/(maxd-mind))*255
            
            # Write the image to the file path
            tifffile.imwrite(file_path_1,depth)
            imageio.imwrite(file_path_2,depth2)

            print('Low Resolution Depth Image Captured')

########################################### Left/Right Low Res Image Capture ##################################################

            # Grab an image, a RuntimeParameters object must be given to grab()

            zed.retrieve_image(image_zed,sl.VIEW.LEFT)
            left = cv2.cvtColor(image_zed.get_data(), cv2.COLOR_BGR2RGB)
            left = cv2.resize(left, (640,360), cv2.INTER_LANCZOS4)
            

            # Find the number of images in the specified directory
            total_images = np.size(os.listdir('/home/shane/final_python_images/low_light/low_left_tif'))

            # Increase the image count by one
            add_image = total_images + 1

            # Append the image counter to the image name
            new_image_num = str(add_image)
            image_name_1 = "image_"+new_image_num+'.tif'
            image_name_2 = "image_"+new_image_num+'.png'

            # Form the final file path to save the new image to
            file_path_left_1 = os.path.join('/home/shane/final_python_images/low_light/low_left_tif',image_name_1)
            file_path_left_2 = os.path.join('/home/shane/final_python_images/low_light/low_left_png',image_name_2)
            left2 = np.divide(np.sum(left,2),3)
            left2 = np.asarray(left2, dtype=np.float32)
            # Write the image to the file path
            tifffile.imwrite(file_path_left_1,left2)
            imageio.imwrite(file_path_left_2,left)
        
            print('Low Resolution Left Image Captured')

            #####################Right Image Capture#######################

            # Saving the image
            zed.retrieve_image(image_zed,sl.VIEW.RIGHT)
            right = cv2.cvtColor(image_zed.get_data(), cv2.COLOR_BGR2RGB)
            right = cv2.resize(right, (640,360), cv2.INTER_LANCZOS4)
            

            # Find the number of images in the specified directory
            total_images = np.size(os.listdir('/home/shane/final_python_images/low_light/low_right_tif'))
            # Increase the image count by one
            add_image = total_images + 1
            # Append the image counter to the image name
            new_image_num = str(add_image)
            image_name_1 = "image_"+new_image_num+'.tif'
            image_name_2 = "image_"+new_image_num+'.png'

            # Form the final file path to save the new image to
            file_path_right_1 = os.path.join('/home/shane/final_python_images/low_light/low_right_tif',image_name_1)
            file_path_right_2 = os.path.join('/home/shane/final_python_images/low_light/low_right_png',image_name_2)
            right2 = np.divide(np.sum(right,2),3)
            right2 = np.asarray(right2, dtype=np.float32)
            # Write the image to the file path
            tifffile.imwrite(file_path_right_1,right2)
            imageio.imwrite(file_path_right_2,right)

            print('Low Resolution Right Image Captured')


    # Close the camera
    zed.close()    
     
######################################### TURN THE LIGHTS ON ###################################################################
    input_anything = input('Turn Lights on and Press "Enter"') 
    print('Continuing image capture')
########################################### High Def Depth Image Capture ##################################################
# Create a Camera object
    zed = sl.Camera()

    #Set initial parameters
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL  # Use NEURAL depth mode
    init_params.coordinate_units = sl.UNIT.METER  # Use meter units (for depth measurements)
    init_params.camera_resolution = sl.RESOLUTION.HD720 #Using 720 our resolution is 1260x720

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    # Create a greyscale sl.Mat object
    image_zed = sl.Mat(zed.get_camera_information().camera_resolution.width, zed.get_camera_information().camera_resolution.height, sl.MAT_TYPE.F32_C1)

    if zed.grab() == sl.ERROR_CODE.SUCCESS :

            # Retrieve the normalized depth image
            zed.retrieve_measure(image_zed, sl.MEASURE.DEPTH)
            
            # Use get_data() to get the numpy array both for manipulation and for image extraction
            depth = image_zed.get_data()
            depth = cv2.resize(depth, (640,360), cv2.INTER_LANCZOS4)
            # Find the number of images in the specified directory
            total_images = np.size(os.listdir('/home/shane/final_python_images/high_light/high_depth_tif'))
            # Increase the image count by one
            add_image = total_images + 1
            # Append the image counter to the image name
            new_image_num = str(add_image)

            #.Tiff file path
            image_name_1 = "image_"+new_image_num+'.tif'
            image_name_2 = "image_"+new_image_num+'.png'
            # Form the final file path to save the new image to
            file_path_1 = os.path.join('/home/shane/final_python_images/high_light/high_depth_tif',image_name_1)
            file_path_2 = os.path.join('/home/shane/final_python_images/high_light/high_depth_png',image_name_2)
            maxd = 20
            mind = 0.5
            depthinterm = np.asarray(depth, dtype=np.uint8)
            depth2 = ((depthinterm - mind)/(maxd-mind))*255
            # Write the image to the file path
            tifffile.imwrite(file_path_1,depth)
            imageio.imwrite(file_path_2,depth2)
    
            print('High Resolution Depth Image Captured')
           
########################################### Left/Right High Res Image Capture ##################################################

            zed.retrieve_image(image_zed,sl.VIEW.LEFT)
            left = cv2.cvtColor(image_zed.get_data(), cv2.COLOR_BGR2RGB)
            left = cv2.resize(left, (640,360), cv2.INTER_LANCZOS4)
            
            # Find the number of images in the specified directory
            total_images = np.size(os.listdir('/home/shane/final_python_images/high_light/high_left_tif'))
        
            # Increase the image count by one
            add_image = total_images + 1
        
            # Append the image counter to the image name
            new_image_num = str(add_image)
            image_name_1 = "image_"+new_image_num+'.tif'
            image_name_2 = "image_"+new_image_num+'.png'
        
            # Form the final file path to save the new image to
            file_path_left_1 = os.path.join('/home/shane/final_python_images/high_light/high_left_tif',image_name_1)
            file_path_left_2 = os.path.join('/home/shane/final_python_images/high_light/high_left_png',image_name_2)

            left2 = np.divide(np.sum(left,2),3)
            left2 = np.asarray(left2, dty# Create a Camera object
    zed = sl.Camera()

    #Set initial parameters
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL  # Use NEURAL depth mode
    init_params.coordinate_units = sl.UNIT.METER  # Use meter units (for depth measurements)
    init_params.camera_resolution = sl.RESOLUTION.HD720 #Using 720 our resolution is 1260x720

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    # Create a greyscale sl.Mat object
    image_zed = sl.Mat(zed.get_camera_information().camera_resolution.width, zed.get_camera_information().camera_resolution.height, sl.MAT_TYPE.F32_C1)

    if zed.grab() == sl.ERROR_CODE.SUCCESS :

            # Retrieve the normalized depth image
            zed.retrieve_measure(image_zed, sl.MEASURE.DEPTH)
            
            # Use get_data() to get the numpy array both for manipulation and for image extraction
            depth = image_zed.get_data()
            depth = cv2.resize(depth, (640,360), cv2.INTER_LANCZOS4)
            # Find the number of images in the specified directory
            total_images = np.size(os.listdir('/home/shane/final_python_images/high_light/high_depth_tif'))
            # Increase the image count by one
            add_image = total_images + 1
            # Append the image counter to the image name
            new_image_num = str(add_image)

            #.Tiff file path
            image_name_1 = "image_"+new_image_num+'.tif'
            image_name_2 = "image_"+new_image_num+'.png'
            # Form the final file path to save the new image to
            file_path_1 = os.path.join('/home/shane/final_python_images/high_light/high_depth_tif',image_name_1)
            file_path_2 = os.path.join('/home/shane/final_python_images/high_light/high_depth_png',image_name_2)
            maxd = 20
            mind = 0.5
            depthinterm = np.asarray(depth, dtype=np.uint8)
            depth2 = ((depthinterm - mind)/(maxd-mind))*255
            # Write the image to the file path
            tifffile.imwrite(file_path_1,depth)
            imageio.imwrite(file_path_2,depth2)
    
            print('High Resolution Depth Image Captured')ft_2,left)

            print('High Resolution Left Image Captured')

            # Saving the image
            zed.retrieve_image(image_zed,sl.VIEW.RIGHT)
            right = cv2.cvtColor(image_zed.get_data(), cv2.COLOR_BGR2RGB)
            right = cv2.resize(right, (640,360), cv2.INTER_LANCZOS4)
            
            
            # Find the number of images in the specified directory
            total_images = np.size(os.listdir('/home/shane/final_python_images/high_light/high_right_tif'))
            
            # Increase the image count by one
            add_image = total_images + 1
        
            # Append the image counter to the image name
            new_image_num = str(add_image)
            image_name_1 = "image_"+new_image_num+'.tif'
            image_name_2 = "image_"+new_image_num+'.png'

            # Form the final file path to save the new image to
            file_path_right_1 = os.path.join('/home/shane/final_python_images/high_light/high_right_tif',image_name_1)
            file_path_right_2 = os.path.join('/home/shane/final_python_images/high_light/high_right_png',image_name_2)
            right2 = np.divide(np.sum(right,2),3)
            right2 = np.asarray(right2, dtype=np.float32)
            # Write the image to the file path
            tifffile.imwrite(file_path_right_1,right2)
            imageio.imwrite(file_path_right_2,right)

            print('High Resolution Right Image Captured')


    # Close the camera
    zed.close()    


    # Loading images which were captures to ensure all of the data types, resolutions, and matrices align with what
    # we are receiving in ROS

    Ldepthpng = imageio.imread('/home/shane/final_python_images/low_light/low_depth_png/image_1.png')
    Ldepthtiff = imageio.imread('/home/shane/final_python_images/low_light/low_depth_tif/image_1.tif')
    Lleftpng = imageio.imread('/home/shane/final_python_images/low_light/low_left_png/image_1.png')
    Llefttiff = imageio.imread('/home/shane/final_python_images/low_light/low_left_tif/image_1.tif')
    Lrightpng = imageio.imread('/home/shane/final_python_images/low_light/low_right_png/image_1.png')
    Lrighttiff = imageio.imread('/home/shane/final_python_images/low_light/low_right_tif/image_1.tif')

    Hdepthpng = imageio.imread('/home/shane/final_python_images/high_light/high_depth_png/image_1.png')
    Hdepthtiff = imageio.imread('/home/shane/final_python_images/high_light/high_depth_tif/image_1.tif')
    Hleftpng = imageio.imread('/home/shane/final_python_images/high_light/high_left_png/image_1.png')
    Hlefttiff = imageio.imread('/home/shane/final_python_images/high_light/high_left_tif/image_1.tif')
    Hrightpng = imageio.imread('/home/shane/final_python_images/high_light/high_right_png/image_1.png')
    Hrighttiff = imageio.imread('/home/shane/final_python_images/high_light/high_right_tif/image_1.tif')

    a = np.shape(Ldepthpng)
    aa = np.size(Ldepthpng)
    aaa = Ldepthpng.dtype

    b = np.shape(Ldepthtiff)
    bb = np.size(Ldepthtiff)
    bbb = Ldepthtiff.dtype

    c = np.shape(Lleftpng)
    cc = np.size(Lleftpng)
    ccc = Lleftpng.dtype

    d = np.shape(Llefttiff)
    dd = np.size(Llefttiff)
    ddd = Llefttiff.dtype

    e = np.shape(Lrightpng)
    ee = np.size(Lrightpng)
    eee = Lrightpng.dtype

    f = np.shape(Lrighttiff)
    ff = np.size(Lrighttiff)
    fff = Lrighttiff.dtype

    g = np.shape(Hdepthpng)
    gg = np.size(Hdepthpng)
    ggg = Hdepthpng.dtype

    h = np.shape(Hdepthtiff)
    hh = np.size(Hdepthtiff)
    hhh = Hdepthtiff.dtype

    i = np.shape(Hleftpng)
    ii = np.size(Hleftpng)
    iii = Hleftpng.dtype

    j = np.shape(Hlefttiff)
    jj = np.size(Hlefttiff)
    jjj = Hlefttiff.dtype

    k = np.shape(Hrightpng)
    kk = np.size(Hrightpng)
    kkk = Hrightpng.dtype

    l = np.shape(Hrighttiff)
    ll = np.size(Hrighttiff)
    lll = Hrighttiff.dtype
    
    
    Ldepthtiff[Ldepthtiff == -inf] = np.nan
    Ldepthtiff[Ldepthtiff == inf] = np.nan
    maxdepth = np.nanmax(Ldepthtiff)    
    mindepth = np.nanmin(Ldepthtiff)
    
    # Print the results
    print("Low Depth PNG has a shape of:",a ,"a size of:",aa ,"and a data type of:",aaa,'\n',
    "Low Depth TIF has a shape of:",b ,"a size of:",bb ,"and a data type of:",bbb ,'\n',
    "Low Left PNG has a shape of:", c,"a size of:", cc,"and a data type of:",ccc ,'\n',
    "Low Left TIF has a shape of:",d ,"a size of:",dd ,"and a data type of:",ddd ,'\n',
    "Low Right PNG has a shape of:", e,"a size of:", ee,"and a data type of:",eee ,'\n',
    "Low Right TIF has a shape of:", f,"a size of:",ff ,"and a data type of:",fff ,'\n',
    "High Depth PNG has a shape of:", g,"a size of:",gg ,"and a data type of:",ggg ,'\n',
    "High Depth TIF has a shape of:",h ,"a size of:", hh,"and a data type of:",hhh ,'\n',
    "High Left PNG has a shape of:", i,"a size of:",ii ,"and a data type of:",iii ,'\n',
    "High Left TIF has a shape of:", j,"a size of:", jj,"and a data type of:",jjj ,'\n',
    "High Right PNG has a shape of:", k,"a size of:",kk ,"and a data type of:", kkk,'\n',
    "High Right TIF has a shape of:", l,"a size of:", ll,"and a data type of:",lll,'\n'
    "Depth max is:",maxdepth,"Depth min is:",mindepth)

if __name__ == "__main__":
    main()


