cmake_minimum_required(VERSION 3.10)
# 设置工程名称
set(PROJECT_NAME cuda_learning)
 
# 设置工程版本号
set(PROJECT_VERSION "0.0.1" CACHE STRING "default")

# 工程定义
project(${PROJECT_NAME}
    LANGUAGES CXX C CUDA
    VERSION ${PROJECT_VERSION}
)
set(CMAKE_CUDA_STANDARD 17)
find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3; -gencode arch=compute_86,code=sm_86)

option(BUILD_SAMPLES "build samples" ON)
option(BUILD_TESTS "build tests" OFF)

IF(BUILD_SAMPLES)
    ADD_SUBDIRECTORY(src)
ENDIF()

include(/usr/local/libtorch/share/cmake/Torch/TorchConfig.cmake)
find_package(Torch)
IF (Torch_FOUND)
    ADD_SUBDIRECTORY(with_torch)
ENDIF ()


# CMAKE_INSTALL_PREFIX的默认定义是/usr/local
INSTALL(DIRECTORY ${PROJECT_SOURCE_DIR}/src/ DESTINATION src USE_SOURCE_PERMISSIONS)
