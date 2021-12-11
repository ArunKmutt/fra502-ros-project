cmake_minimum_required(VERSION 3.0.2)
project(gh_gazebo)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  cv_bridge
  sensor_msgs
)
find_package(OpenCV REQUIRED)

catkin_package(

)

install(DIRECTORY
        launch
        world
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
