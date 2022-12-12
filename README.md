# Depth image enhancement for UAV exploration

**A project within the class "ECE 740 - Computer and Robot Vision" at University of Alberta.** </br> 
Shane Allan, Sophia Wagner, Almas Sahar

In this work, an enhanced depth estimation methodology is proposed using a low-cost stereo-vision camera that can deliver quality depth images in low-light environments suitable for real-time implementation. To address this problem, a convolutional neural network (CNN) is recommended to enhance the raw stereo red, green, and blue (RGB) and depth images for use on a prototype unmanned aerial vehicle (UAV). 
Our contributions are the following: 
- Development of different CNN models in order to enhance depth from low-light depth and stereo images.
- Generation of a dataset with varying quality and lighting conditions consisting of depth and stereo images.
- Running the CNN models on the onboard computing module of the UAV. 

The generated dataset can be found [here](https://drive.google.com/drive/folders/1QiNkWCm-5WVvvHuQiLhKXMPIqyYSNP9o?usp=sharing)

--- 

### Resources 

**Software:**
- PyTorch
- Python 2.7, 3.6
- C++
- ROS Melodic
- Ubuntu 18.04 

**Hardware:**
- DJI F450 Flamewheel Frame
- Garmin ToF Laser Rangefinder
- Pixhawk 5x Flight Controller 
- Sereolabs Zed 2 Camera
- Jetson Xavier NX 


--- 

### Results 


Architecture             | Results   
:-------------------------:|:-------------------------:
![](https://github.com/sophiajwagner/depth-image-enhancement-for-UAV-exploration/blob/main/img/model2_arch.png)  |  ![](https://github.com/sophiajwagner/depth-image-enhancement-for-UAV-exploration/blob/main/img/model2_preds.PNG)




Architecture             | Results   
:-------------------------:|:-------------------------:
![](https://github.com/sophiajwagner/depth-image-enhancement-for-UAV-exploration/blob/main/img/model3_arch.png)  |  ![](https://github.com/sophiajwagner/depth-image-enhancement-for-UAV-exploration/blob/main/img/model3_preds.PNG)
