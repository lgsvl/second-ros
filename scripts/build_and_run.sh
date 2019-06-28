#!/bin/bash
export PYTHONUNBUFFERED=1
source /opt/ros/kinetic/setup.bash
cd /root/catkin_ws
catkin_make
source devel/setup.bash
roslaunch second_ros second_ros.launch