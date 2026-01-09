cmake_minimum_required(VERSION 3.10)
project(${PROJECT_NAME}
    LANGUAGES CXX C CUDA
    VERSION ${PROJECT_VERSION}
)
set(CMAKE_CUDA_STANDARD 17)
find_package(CUDA REQUIRED)

find_package(Torch)
IF (Torch_FOUND)
    include_directories(${TORCH_INCLUDE_DIRS})
    add_executable(sum_test sum.cu)
    target_link_libraries(sum_test ${TORCH_LIBRARIES})
    install(TARGETS sum_test DESTINATION install)
ENDIF ()