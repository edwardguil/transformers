cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(train)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_PREFIX_PATH "<libtorch-path>/share/cmake/Torch")

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(SOURCES train.cpp transformer.cpp)  # Add transformer.cpp to the source list
add_executable(train ${SOURCES})
target_link_libraries(train "${TORCH_LIBRARIES}")
set_property(TARGET train PROPERTY CXX_STANDARD 17)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET train
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:train>)
endif (MSVC)
