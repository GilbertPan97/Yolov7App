# CMakeLists.txt
cmake_minimum_required(VERSION 3.12)
project(ipcBackend)

set(DEPS ${CMAKE_CURRENT_SOURCE_DIR}/3rd-party)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

# config onnx build option
set(ONNX_ROOT_DIR ${DEPS}/onnxruntime-linux-x64-gpu-1.12.1)
option(BUILD_EXAMPLE "Build onnxruntime example" ON)
option(WITH_GPU "Inference with cuda" ON)
if(WITH_GPU)
    add_definitions(-DWITH_GPU=1)
endif() 

# Find OpenCV package
find_package(Boost COMPONENTS filesystem system HINTS ${DEPS}/boost_1_82_0)
find_package(Eigen3 REQUIRED HINTS ${DEPS}/eigen)
find_package(OpenCV REQUIRED HINTS ${DEPS}/opencv)
find_package(websocketpp REQUIRED HINTS ${DEPS}/websocketpp-0.8.2)

# Set the output directory to 'export'
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/export/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/export/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/export/bin)
if (MSVC)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_SOURCE_DIR}/export/bin)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_SOURCE_DIR}/export/lib)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_SOURCE_DIR}/export/lib)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/export/bin)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/export/lib)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/export/lib)
endif()

# Set CMAKE_MODULE_PATH for cmake module
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

# Add onnx lib
add_subdirectory(ModelPredict)
add_subdirectory(CvInference)

# Build camera lib
add_library(CameraLib Camera.cpp)
target_include_directories(CameraLib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(CameraLib PUBLIC ${OpenCV_LIBS})
include_directories(${Boost_INCLUDE_DIRS})

# Build websocket lib
add_library(WebSocketServer WebSocketServer.cpp)
target_include_directories(WebSocketServer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(WebSocketServer PUBLIC websocketpp::websocketpp CameraLib CvInference)

# Add source files
add_executable(ipcBackend main.cpp)

# Link against OpenCV
target_link_libraries(ipcBackend PRIVATE WebSocketServer)

add_custom_command(TARGET "ipcBackend" POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/models/yolov8.onnx
    $<TARGET_FILE_DIR:ipcBackend>)
