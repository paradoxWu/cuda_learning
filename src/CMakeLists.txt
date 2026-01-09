cmake_minimum_required(VERSION 3.10)
project(${PROJECT_NAME}
    LANGUAGES CXX C CUDA
    VERSION ${PROJECT_VERSION}
)

set(CMAKE_CUDA_STANDARD 17)
find_package(CUDA REQUIRED)

add_executable(hello_world hello.cu)
add_executable(add_vec vecadd.cu)
add_executable(matmul matmul.cu)
add_executable(mattran mattran.cu)
add_executable(matmul_v3 matmul_v3.cu)
add_executable(reduce reduce.cu)

find_package(OpenCV)
IF (OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
    add_executable(gaussian_blur gaussian_blur.cu )
    target_link_libraries(gaussian_blur ${OpenCV_LIBS} )
    install(TARGETS gaussian_blur DESTINATION install)
    install(DIRECTORY ${PROJECT_SOURCE_DIR}/images DESTINATION ${CMAKE_INSTALL_PREFIX}/install USE_SOURCE_PERMISSIONS)
ENDIF ()




# CMAKE_INSTALL_PREFIX的默认定义是/usr/local
install(TARGETS hello_world DESTINATION install)
install(TARGETS add_vec DESTINATION install)
install(TARGETS matmul DESTINATION install)
install(TARGETS mattran DESTINATION install)
install(TARGETS matmul_v3 DESTINATION install)
install(TARGETS reduce DESTINATION install)

