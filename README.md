# Yolov5_DeepSort_Pytorch_ROS
you can use yolov5 deepsort in ros

I modified the code on the page below to be used in ROS.


https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch


Modified so that parameter input is not required.


For performance, inference starts immediately without going through RGB->BGR conversion.


Just pass the image to "/camera/color/image_raw".


I set the image resolution to 848*480, you need to change this to use it.


Detect data is delivered in the form of BoundingBoxIXYWHArray. node name : detect_results


BoundingBoxIXYWHArray.msg

kanu_msgs/BoundingBoxIXYWH[] boxes


BoundingBoxIXYWH.msg :

int32 i

int32 x

int32 y

int32 w

int32 h
