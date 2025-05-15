cmake_minimum_required(VERSION 3.0)
project(OnnxDeploy CXX C)

set(CMAKE_DEBUG_POSTFIX "d" CACHE STRING "add a postfix, usually d on windows")
set(CMAKE_RELEASE_POSTFIX "" CACHE STRING "add a postfix, usually empty on windows")
set(CMAKE_RELWITHDEBINFO_POSTFIX "rd" CACHE STRING "add a postfix, usually empty on windows")
set(CMAKE_MINSIZEREL_POSTFIX "s" CACHE STRING "add a postfix, usually empty on windows")

# option(WITH_OPENCV "Run onnx model with opencv" OFF)
option(WITH_GPU "Inference with cuda" ON)
option(BUILD_SHARED "Build librarys with dll output" OFF)
option(BUILD_EXAMPLE "Build example" ON)

# Set project dep and output dirs
set(THIRD_PARTY "${CMAKE_CURRENT_SOURCE_DIR}/third_party")
set(OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/version_1.0")
set(PROJECT_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Set CMAKE_MODULE_PATH for cmake module
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/")

# Include project cmake utils
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/OnnxDeployUtils.cmake)

# Set source default output dir
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
add_definitions(-DONNX_DEPLOY)

# Include project source directory
add_subdirectory(src)
