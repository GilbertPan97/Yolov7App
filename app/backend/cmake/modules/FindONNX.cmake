# Set OnnxRuntime version
set(ONNX_VERSION 1.12.1)

if(NOT ONNX_ROOT_DIR)
    set(ONNX_ROOT_DIR ${ORT_DIR})
    message(STATUS "ONNX_ROOT_DIR is: ${ONNX_ROOT_DIR}")
endif()

# For now, check the old name ORT_INSTALL_DIR
if(ORT_INSTALL_DIR AND NOT ONNX_ROOT_DIR)
    set(ONNX_ROOT_DIR ${ORT_INSTALL_DIR})
endif()

if(ONNX_ROOT_DIR)
    find_library(ORT_LIB onnxruntime
        ${ONNX_ROOT_DIR}/lib
        CMAKE_FIND_ROOT_PATH_BOTH)
    get_filename_component(ORT_LIB_DIR "${ORT_LIB}" DIRECTORY)

    if(WIN32)
        file(GLOB ORT_LIBS "${ORT_LIB_DIR}/*.lib")
        file(GLOB ORT_DLLS "${ORT_LIB_DIR}/*.dll")
    elseif(UNIX)
        file(GLOB ORT_LIBS "${ORT_LIB_DIR}/*.so")
    endif()
    
    find_path(ORT_INCLUDE onnxruntime_cxx_api.h
        ${ONNX_ROOT_DIR}/include
        CMAKE_FIND_ROOT_PATH_BOTH)
endif()

# Chech ONNX search status
if(ORT_LIBS AND ORT_INCLUDE)
    set(ONNX_FOUND TRUE)
else()
    set(ONNX_FOUND FALSE)
endif()

# provide found onnxruntime lib and header message
if(ONNX_FOUND)
    # For CMake output only
    set(ONNX_LIBRARIES "${ORT_LIBS}" CACHE STRING "ONNX Runtime libraries")
    set(ONNX_INCLUDE_DIR "${ORT_INCLUDE}" CACHE STRING "ONNX Runtime include path")
    message(STATUS "Found ONNX Runtime: ${ONNX_ROOT_DIR}")

    # Link target with associated interface headers
    set(ONNX_LIBRARY "onnxruntime" CACHE STRING "ONNX Link Target")
    add_library(${ONNX_LIBRARY} SHARED IMPORTED)
    set_target_properties(${ONNX_LIBRARY} PROPERTIES
                          INTERFACE_INCLUDE_DIRECTORIES ${ORT_INCLUDE}
                          IMPORTED_LOCATION ${ORT_LIBS}
                          IMPORTED_IMPLIB ${ORT_LIBS})
    
    # Check whether the library exists, is available, and meets the requirements
    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(ONNX DEFAULT_MSG
        ONNX_INCLUDE_DIR
        ONNX_LIBRARIES
    )
endif()

mark_as_advanced(ONNX_INCLUDE_DIR ONNX_LIBRARIES)
