# CMakeLists.txt
cmake_minimum_required(VERSION 3.12)
project(ipcBackend)

set(DEPS ${CMAKE_CURRENT_SOURCE_DIR}/3rd-party)

# Find OpenCV package
find_package(Boost COMPONENTS filesystem system HINTS ${DEPS}/boost_1_82_0)
find_package(Eigen3 REQUIRED HINTS ${DEPS}/eigen)
find_package(OpenCV REQUIRED HINTS ${DEPS}/opencv)
find_package(websocketpp REQUIRED HINTS ${DEPS}/websocketpp-0.8.2)

# Set the output directory to 'export'
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/export/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/export/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/export/bin)

# Add camera lib
add_library(CameraLib Camera.cpp)
target_include_directories(CameraLib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Camera.h)
target_link_libraries(CameraLib PUBLIC ${OpenCV_LIBS})
include_directories(${Boost_INCLUDE_DIRS})


# Add source files
add_executable(ipcBackend server.cpp)

# Link against OpenCV
target_link_libraries(ipcBackend ${OpenCV_LIBS} CameraLib websocketpp::websocketpp)

