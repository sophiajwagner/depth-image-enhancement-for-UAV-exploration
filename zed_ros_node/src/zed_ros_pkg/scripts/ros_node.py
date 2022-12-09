#!/usr/bin/env python3
import rospy
import message_filters
from sensor_msgs.msg import Image as msg_Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
from numpy import asarray
from numpy import inf

import seaborn as sns
from torchvision.transforms import functional as F
import torch
import torchvision.models as models
import struct
from rospy.numpy_msg import numpy_msg
from numpy import inf
import seaborn as sns
import imageio
import time

import sys
np.set_printoptions(threshold=sys.maxsize)
sys.path.insert(1,'/home/shane/zed_ros_node/src/zed_ros_pkg/scripts/networks/')
import DepthDataset, DepthAutoencoder, train_depth, StereoAutoencoder, StereoDataset, train_stereo
rospy.init_node("zed_ros_node")


class ImageListener:
    def __init__(self):
        
        topicL = '/zed2/zed_node/left_raw/image_raw_color'
        topicR = '/zed2/zed_node/right_raw/image_raw_color'
        topicD = '/zed2/zed_node/depth/depth_registered'
        self.left = message_filters.Subscriber(topicL,numpy_msg(msg_Image))
        self.right = message_filters.Subscriber(topicR,numpy_msg(msg_Image))
        self.depth = message_filters.Subscriber(topicD,numpy_msg(msg_Image))
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        #Model 1 combines the left and right and depths images as inputs to CNN and outputs enhanced Depth
        self.model1=torch.load('/home/shane/zed_ros_node/src/zed_ros_pkg/scripts/weights_threeinputs.pt')
        self.model1.to(self.device)
        self.model1.eval()
        #Model 2 takes in depth image and outputs enhanced depth image
        self.model2=torch.load('/home/shane/zed_ros_node/src/zed_ros_pkg/scripts/weights_depth.pt')
        self.model2.to(self.device)
        self.model2.eval()

        self.rate = rospy.Rate(15)
        self.x = list()
        self.y = list()
        self.i = 0


    def Callback(self,left,right,depth):
        
        #Gamma Filter Definition Ref: Dr. Xin Xin Zuo
        def gammaCorrection(src, gamma):
            invGamma = 1 / gamma
            table = [((i / 255) ** invGamma) * 255 for i in range(256)]
            table = np.array(table, np.uint8)
            return cv2.LUT(src,table)
        
        #Generating numpy arrays from ROS topics
        left_image_4_ch = np.frombuffer(left.data, dtype=np.uint8).reshape(left.height, left.width, -1)
        right_image_4_ch = np.frombuffer(right.data, dtype=np.uint8).reshape(right.height, right.width, -1)
        left_image_4_ch = gammaCorrection(left_image_4_ch, 2)
        right_image_4_ch = gammaCorrection(right_image_4_ch, 2)
        depth_image_raw = np.frombuffer(depth.data, dtype=np.float32).reshape(depth.height, depth.width, -1)
        depth_image_raw = depth_image_raw*1.0
        
        #Removing Infinities and Nan
        depth_image_raw[np.isnan(depth_image_raw)] = 0
        depth_image_raw[depth_image_raw == -np.inf] = 0.5
        depth_image_raw[depth_image_raw == np.inf] = 20
       
        #Normalize Input between 0 and 1 using min max
        depth_image = ((depth_image_raw-0.5)/19.5)
        
        #Display Original Depth Image
        cv2.imshow("Original Depth Image",depth_image)
        
        #Conversion to 32 bit Floats
        left_image = left_image_4_ch[:,:,:3]/255
        left_image = np.divide(np.sum(left_image,2),3)
        left_image = np.asarray(left_image, dtype=np.float32)
        right_image = right_image_4_ch[:,:,:3]/255
        right_image = np.divide(np.sum(right_image,2),3)
        right_image = np.asarray(right_image, dtype=np.float32)

        #Model #1
        imgL = torch.from_numpy(left_image).to(self.device).unsqueeze(0).unsqueeze(0).float()
        imgR = torch.from_numpy(right_image).to(self.device).unsqueeze(0).unsqueeze(0).float()
        imgD = torch.from_numpy(depth_image).to(self.device).unsqueeze(0).float()
        imgD = torch.permute(imgD,(0,3,1,2))
        model1_out = self.model1(imgL, imgR, imgD).squeeze().cpu().detach().numpy()
        
        #Displaying Model #1 Output
        cv2.imshow("Depth Output first model",model1_out)

        #Model #2
        depth_image_proc = torch.from_numpy(depth_image).to(self.device).unsqueeze(0).float()
        depth_image_proc = torch.permute(depth_image_proc,(0,3,1,2))
        model2_out = self.model2(depth_image_proc).squeeze().cpu().detach().numpy()
        
        #Displaying Model #2 Output
        cv2.imshow("Enhanced Depth from Model 2",model2_out)   

        #Retrieving actual depth values after normalization
        model2_out = (model2_out*19.5)+0.5
        
        #Calculating original depth at center pixel
        original_depth = depth_image_raw[int(len(depth_image_raw)/2)][int(len(depth_image_raw[0])/2)]
        
        #Calculating depth at center pixel from model output
        enhanced_depth = model2_out[int(len(model2_out)/2)][int(len(model2_out[0])/2)]
    
        #Print the calculated depths
        print("Original Depth:",original_depth,"Depth Estimated from Model:",enhanced_depth)
        
        #Print average depths after 50, 100, and 150 frames to guage accuracy
        if original_depth != 0 and original_depth <10:
            self.x.append(original_depth)
          
        if enhanced_depth != 0 and enhanced_depth<10:
            self.y.append(enhanced_depth)

        if self.i == 50 or self.i == 100 or self.i == 150:
            print("OD",np.average(self.x)-6)
            print("ED",np.average(self.y))

        self.i=self.i+1

        cv2.waitKey(3)

if __name__ == '__main__':
    listener = ImageListener()
    #rospy.init_node("zed_ros_node")
    try:
        ts = message_filters.ApproximateTimeSynchronizer([listener.left,listener.right, listener.depth],10,0.1)
        ts.registerCallback(listener.Callback)
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

