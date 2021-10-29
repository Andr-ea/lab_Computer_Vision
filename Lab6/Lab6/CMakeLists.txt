cmake_minimum_required(VERSION 2.8)

project(main)

#set(OpenCV_DIR /usr/local/Cellar/opencv@3/3.4.9_2/share/OpenCV)

find_package(OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(${PROJECT_NAME} src/main.cpp )

target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )