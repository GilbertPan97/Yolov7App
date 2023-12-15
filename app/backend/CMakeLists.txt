# CMakeLists.txt
cmake_minimum_required(VERSION 3.12)
project(ipcBackend)

set(DEPS ${CMAKE_CURRENT_SOURCE_DIR}/3rd-party)

# config onnx build option
set(ONNX_ROOT_DIR ${DEPS}/onnxruntime-linux-x64-gpu-1.12.1)
option(BUILD_EXAMPLE "Build onnxruntime example" ON)
option(WITH_GPU "Inference with cuda" ON)
if(WITH_GPU)
    add_definitions(-DWITH_GPU=1)
endif() 

# Find OpenCV package
find_package(Eigen3 REQUIRED HINTS ${DEPS}/eigen)
find_package(OpenCV REQUIRED HINTS ${DEPS}/opencv)
find_package(websocketpp REQUIRED HINTS ${DEPS}/websocketpp-0.8.2)

# Set the output directory to 'export'
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/export/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/export/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/export/bin)

# Set CMAKE_MODULE_PATH for cmake module
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

# Add onnx lib
add_subdirectory(ModelPredict)

# Build camera lib
add_library(CameraLib Camera.cpp)
target_include_directories(CameraLib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(CameraLib PUBLIC ${OpenCV_LIBS})

# Build websocket lib
add_library(WebSocketServer WebSocketServer.cpp)
target_include_directories(WebSocketServer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(WebSocketServer PUBLIC websocketpp::websocketpp CameraLib ModelPredict)

# Add source files
add_executable(ipcBackend main.cpp)

# Link against OpenCV
target_link_libraries(ipcBackend PRIVATE WebSocketServer)
