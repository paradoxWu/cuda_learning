cmake_minimum_required(VERSION 3.10)
project(${PROJECT_NAME}
    LANGUAGES CXX C CUDA
    VERSION ${PROJECT_VERSION}
)

set(CMAKE_CUDA_STANDARD 11)

add_executable(hello_world hello.cu)
# set_target_properties(
#         hello_world
#         PROPERTIES
#         CUDA_SEPARABLE_COMPILATION ON)

add_executable(add_vec vecadd.cu)

find_package( OpenCV REQUIRED )
IF (OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
ELSE ()
    ADD_DEFINITIONS(-DWITHOUT_CV)
ENDIF ()
add_executable(gaussian_blur gaussian_blur.cu )
target_link_libraries(gaussian_blur ${OpenCV_LIBS} )



# CMAKE_INSTALL_PREFIX的默认定义是/usr/local
install(TARGETS hello_world DESTINATION install)
install(TARGETS add_vec DESTINATION install)
install(TARGETS gaussian_blur DESTINATION install)