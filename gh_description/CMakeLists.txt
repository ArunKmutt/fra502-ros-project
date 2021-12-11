cmake_minimum_required(VERSION 3.0.2)
project(gh_description)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
)

catkin_package(

)

install(DIRECTORY
        rviz
        meshes 
        launch
        urdf
        world
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
