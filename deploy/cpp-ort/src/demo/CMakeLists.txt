
set(SRC_FILES demo_main.cpp)

# generate project target
add_executable(demo_mp ${SRC_FILES})

target_link_libraries(demo_mp PUBLIC 
                      ${ONNX_LIBRARIES} ${OpenCV_LIBS})

target_link_libraries(demo_mp PRIVATE ${LIBS_NAME})

target_include_directories(demo_mp PUBLIC 
                           ${ONNX_INCLUDE_DIR}
                           ${OpenCV_INCLUDE_DIRS})

target_include_directories(demo_mp PRIVATE 
            "${CMAKE_CURRENT_SOURCE_DIR}/..")

# The onnxruntime DLLs need to be copied to avoid memory errors.
if (MSVC)
    add_custom_command(TARGET "demo_mp" POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy_if_different
                       ${ORT_DLLS}
                       $<TARGET_FILE_DIR:demo_mp>)
endif (MSVC)