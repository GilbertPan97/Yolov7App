set(LIBS_NAME ModelPredict)

# opencv is necessory
if (WIN32)
  find_package(OpenCV REQUIRED HINTS ${THIRD_PARTY}/opencv NO_DEFAULT_PATH)
else ()
  if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64") # x86_64 aarch64
    find_package(OpenCV REQUIRED)
    # set(OpenCV_INCLUDE_DIRS "/usr/include/opencv4")
    # file(GLOB OpenCV_LIBS /usr/lib/aarch64-linux-gnu/libopencv_*${CMAKE_SHARED_LIBRARY_SUFFIX})
  else()
    find_package(OpenCV REQUIRED )
  endif()
endif ()

# Add ort dependence
config_onnx_path("onnxruntime" "1.18.1" ONNX_ROOT_DIR)
find_package(ONNX REQUIRED)
message(STATUS "Find package status is: ${ONNX_FOUND}")

# # Find the OpenSSL package
find_package(OpenSSL REQUIRED)
# # Find the OpenSSL package
# set(OPENSSL_ROOT_DIR ${THIRD_PARTY}/OpenSSL-Win64)
# set(OPENSSL_INCLUDE_DIR ${OPENSSL_ROOT_DIR}/include)
# set(OPENSSL_LIBRARIES "${OPENSSL_ROOT_DIR}/lib/VC/x64/MD/libssl.lib;
#   ${OPENSSL_ROOT_DIR}/lib/VC/x64/MD/libcrypto.lib")
cmake_policy(SET CMP0004 OLD)

# config ModelPredict lib build
set(SRC_FILES ModelPredict.cpp)

set(HEADER_FILES 
    ModelPredict.h
    ModelPredictGlobal.h
)

source_group("Source Files" FILES ${SRC_FILES})
source_group("Header Files" FILES ${HEADER_FILES})

set(CMAKE_INCLUDE_CURRENT_DIR ON)

if(BUILD_SHARED)
  add_library(${LIBS_NAME} SHARED ${SRC_FILES} ${HEADER_FILES})
  target_compile_definitions(${LIBS_NAME} PRIVATE MP_SHARED_EXPORT)
else()
  add_library(${LIBS_NAME} STATIC ${SRC_FILES} ${HEADER_FILES})
  target_compile_definitions(${LIBS_NAME} PUBLIC MP_STATIC)  
endif()

# Add openssl deps
target_link_libraries(${LIBS_NAME} PUBLIC ${OpenCV_LIBS} ${ONNX_LIBRARIES} ${OPENSSL_LIBRARIES})

target_include_directories(${LIBS_NAME} PUBLIC 
                      ${OpenCV_INCLUDE_DIRS} 
                      ${ONNX_INCLUDE_DIR}
                      ${OPENSSL_INCLUDE_DIR}
                      ${CMAKE_CURRENT_SOURCE_DIR})

set_target_properties(${LIBS_NAME} PROPERTIES
                      CXX_STANDARD 17
                      CXX_STANDARD_REQUIRED ON)

# Add demo subdirectory
if(BUILD_EXAMPLE)
  add_subdirectory(demo)
endif()