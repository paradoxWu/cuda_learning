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
set(CMAKE_CUDA_STANDARD 11)


option(BUILD_SAMPLES "build samples" ON)
option(BUILD_TESTS "build tests" OFF)
# add_executable(${PROJECT_NAME}_hello_world samples/hello.cu)


IF(BUILD_SAMPLES)
    ADD_SUBDIRECTORY(src)
ENDIF()

# set_target_properties(
#         cuda_01_hello_world
#         PROPERTIES
#         CUDA_SEPARABLE_COMPILATION ON)

# CMAKE_INSTALL_PREFIX的默认定义是/usr/local
INSTALL(DIRECTORY ${PROJECT_SOURCE_DIR}/src/ DESTINATION src USE_SOURCE_PERMISSIONS)
INSTALL(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/images DESTINATION ${CMAKE_INSTALL_PREFIX}/install USE_SOURCE_PERMISSIONS)