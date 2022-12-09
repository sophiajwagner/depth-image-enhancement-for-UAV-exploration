
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <opencv2/core/base.hpp>
#include <opencv2/imgcodecs.hpp>
#include <sstream>
#include <string>
#include <opencv2/core/types.hpp>
using namespace std;

static const std::string OPENCV_WINDOW = "Image window";

class ImageConverter
{
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  image_transport::Publisher image_pub_;
  

  public:
  ImageConverter()
    : it_(nh_)
  {
    // Subscrive to input video feed and publish output video feed
    image_sub_ = it_.subscribe("/zed2/zed_node/depth/depth_registered", 1, &ImageConverter::imageCb, this);
    image_pub_ = it_.advertise("/image_converter/output_video", 1);

    cv::namedWindow(OPENCV_WINDOW);
  }

  ~ImageConverter()
  {
    cv::destroyWindow(OPENCV_WINDOW);
  }
  
  void imageCb(const sensor_msgs::ImageConstPtr& msg)
  {
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
      
     cv_ptr = cv_bridge::toCvCopy(msg,sensor_msgs::image_encodings::TYPE_32FC1);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }

    cv::Mat depth = cv_ptr->image;
    
    int rows = depth.rows;
    int cols = depth.cols;
    //cout << rows;
    //cout<< cols;
    //cv::Size s = depth.size();
    //cout << rows = s.height;
    //cout << cols = s.width;
    //cv::waitKey(0);
    //cout << depth;
    //cv::convertTo::
    cv::Mat norm_img;
    
    cv::normalize(depth, norm_img, 0,1,cv::NORM_MINMAX,CV_32FC1);
    //cout <<norm_img;
    // Update GUI Window
    //cv::imshow(OPENCV_WINDOW,depth);
    //cv::waitKey(0);
    //print(cv_ptr);
    //cout << depth;
    //cv::ImreadModes IMWRITE_TIFF_COMPRESSION;
    cv::imwrite("/home/shane/python_images/image_1.tif", depth);
    //image_sub_.shutdown();
    //cv::waitKey(0);

    //static int count = 0;
    //ROS_ASSERT( cv::imwrite( std::string( "/home/shane/python_images/image_" ) + std::string( ".tif" ), depth ) );
    //count = count + 1;
    




    // Output modified video stream
    //image_pub_.publish(cv_ptr->toImageMsg());
  }
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "image_converter");
  ImageConverter ic;
  ros::spin();
  return 0;
}