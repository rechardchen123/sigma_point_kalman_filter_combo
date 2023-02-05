cmake_minimum_required(VERSION 3.5)
project(spkf)

# Set build type to Release if not explicitly specified
IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Release)
ENDIF()
MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

# Set C++ compiler flags
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -pedantic -Wextra -fmessage-length=0 -O2 -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic -Wextra -fmessage-length=0 -O2 -march=native")

# Check C++14 compiler support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX14)
if(COMPILER_SUPPORTS_CXX14)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
    add_definitions(-DCOMPILEDWITHC14)
    message(STATUS "Using flag -std=c++14.")
else()
    message(FATAL_ERROR "${CMAKE_CXX_COMPILER} does not support C++14 standard.")
endif()


# Find required packages
find_package(Eigen3 REQUIRED)

# Include paths
include_directories(${PROJECT_SOURCE_DIR}/sigma_point_kalman_filter/
                    ${EIGEN3_INCLUDE_DIR})


# Build examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
add_executable(unicycle examples/unicycle.cpp spkf/kf_base.h spkf/ekf.h spkf/sigma_base.h spkf/ukf.h spkf/cdkf.h spkf/sqrt_base.h spkf/sqrt_ukf.h spkf/sqrt_cdkf.h)