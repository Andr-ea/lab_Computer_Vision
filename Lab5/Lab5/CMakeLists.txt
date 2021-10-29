cmake_minimum_required(VERSION 2.8)
project(Main)

#set(OpenCV_DIR /usr/local/Cellar/opencv@3/3.4.9_2/share/OpenCV)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} src/Main.cpp src/PanoramicImage.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
target_include_directories(${PROJECT_NAME} PRIVATE include)