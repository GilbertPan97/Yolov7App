# function: FetchAndExtractDependency
function(FetchAndExtractDependency DEPENDENCY_NAME DEPENDENCY_URL DEPENDENCY_PREFIX HASH_SHA256)
    include(ExternalProject)
    message(STATUS "Downloading and extracting ${DEPENDENCY_NAME} Setting...")

    # add ExternalProject
    ExternalProject_Add( ${DEPENDENCY_NAME}
        PREFIX                      ${DEPENDENCY_PREFIX}
        DOWNLOAD_EXTRACT_TIMESTAMP  TRUE
        UPDATE_DISCONNECTED         FALSE
        URL                         ${DEPENDENCY_URL}
        URL_HASH                    ${HASH_SHA256}
        DOWNLOAD_NO_PROGRESS        1
        CONFIGURE_COMMAND           ""
        BUILD_COMMAND               ""
        INSTALL_COMMAND             ""
    )

    message(STATUS "Download and extraction of ${DEPENDENCY_NAME} setting completed.")
endfunction()

# Set OnnxRuntime version
set(ONNX_VERSION 1.12.1)

if(ONNX_ROOT_DIR AND EXISTS ${ONNX_ROOT_DIR})
    message(STATUS "ONNX_ROOT_DIR is: ${ONNX_ROOT_DIR}")
else()
    message(STATUS "The dependency of OnnxRuntime is not exists.")
    set(OnnxRuntime "OnnxRuntime")
    set(DEPENDENCY_URL https://github.com/microsoft/onnxruntime/releases/download/v1.12.1/onnxruntime-win-x64-1.12.1.zip)
    set(DEPENDENCY_PREFIX ${CMAKE_BINARY_DIR}/OnnxRuntime)
    set(DEPENDENCY_HASH SHA256=c69650ba14aeae5903b05256a82e77164fff2de992072bc695a3838c1830b85a)
    # call FetchAndExtractDependency
    FetchAndExtractDependency(OnnxRuntime ${DEPENDENCY_URL} ${DEPENDENCY_PREFIX} ${DEPENDENCY_HASH})
    set(ONNX_ROOT_DIR "${DEPENDENCY_PREFIX}/src/OnnxRuntime")
    message(STATUS "Redirect OnnxRuntime directory to: ${ONNX_ROOT_DIR}")
endif()

# Check the Onnxruntime installed dir ${ORT_INSTALL_DIR}
if(ORT_INSTALL_DIR AND NOT ONNX_ROOT_DIR)
    set(ONNX_ROOT_DIR ${ORT_INSTALL_DIR})
endif()

# Config find_package variables
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
