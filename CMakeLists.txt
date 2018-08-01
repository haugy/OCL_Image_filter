project(ocl_test)

#cmake version
cmake_minimum_required(VERSION 3.5)

#add_executable(ocl_test ${CMAKE_SOURCE_DIR}/ocl_test.cpp)
#add_executable(ocl_run_test ${CMAKE_SOURCE_DIR}/ocl_demo_test.cpp)
add_executable(ocl_sobel ${CMAKE_SOURCE_DIR}/ocl_filter.cpp)

#link lib
find_package(OpenCL)
if (OpenCL_FOUND)
    include_directories(${OpenCL_INCLUDE_DIRS})
    #target_link_libraries(ocl_test ${OpenCL_LIBRARIES})
    #target_link_libraries(ocl_run_test ${OpenCL_LIBRARIES})
    target_link_libraries(ocl_sobel ${OpenCL_LIBRARIES})
endif (OpenCL_FOUND)

#OpenCV lib
find_package(OpenCV)
if (OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
    target_link_libraries(ocl_sobel ${OpenCV_LIBRARIES})
endif (OpenCV_FOUND)
