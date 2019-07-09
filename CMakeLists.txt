cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(LoadModel)

find_package(OpenCV REQUIRED PATHS "D:/ThirdParty_MT/opencv/build")
find_package(Torch REQUIRED PATHS "D:/ThirdParty_MT/libtorch")

add_executable(LoadModel main.cpp)
include_directories(${OpenCV_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS})
target_link_libraries(LoadModel "${OpenCV_LIBS}" "${TORCH_LIBRARIES}")
set_property(TARGET LoadModel PROPERTY CXX_STANDARD 11)