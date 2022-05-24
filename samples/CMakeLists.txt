cmake_minimum_required(VERSION 3.16)
project(${PROJECT_NAME}
    LANGUAGES CXX C CUDA
    VERSION ${PROJECT_VERSION}
)

set(CMAKE_CUDA_STANDARD 11)

add_executable(${PROJECT_NAME}_hello_world hello.cu)
add_executable(${PROJECT_NAME}_add_vec vecadd.cu)

set_target_properties(
        ${PROJECT_NAME}_hello_world
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

# CMAKE_INSTALL_PREFIX的默认定义是/usr/local
install(TARGETS ${PROJECT_NAME}_hello_world DESTINATION install)
install(TARGETS ${PROJECT_NAME}_add_vec DESTINATION install)