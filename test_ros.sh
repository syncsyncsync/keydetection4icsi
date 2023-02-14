#!/bin/bash
NodeName="sperm_test_image"
IMAGE="test.png"
roscore &
sleep 2
#rosrun image_folder_publisher image_folder_publisher.py  _image_folder:=/home/icsiauto/detectron2/tst_images _topic_name:=$NodeName &
rosrun image_publisher image_publisher $IMAGE __name:=$NodeName &
sleep 2
echo "roscore and image_publisher are running"
#rosrun image_view image_view image:=/$NodeName/image __name:=$NodeName"_view" &

