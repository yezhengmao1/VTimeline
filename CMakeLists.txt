cmake_minimum_required(VERSION 3.22)

project(vtimeline LANGUAGES CXX)
set(LIBNAME vtimeline)

# specify the c++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# generate compile_commands for code analyze
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDA REQUIRED)
find_package(spdlog REQUIRED)

set(SOURCES vtimeline.cpp)

add_library(${LIBNAME} SHARED ${SOURCES})

set_target_properties(
    ${LIBNAME} PROPERTIES
    VERSION 0.0.1
)

target_include_directories(
    ${LIBNAME}
    PUBLIC
    PRIVATE
    ${CUDA_TOOLKIT_ROOT_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/
)

message(STATUS "${CMAKE_CURRENT_BINARY_DIR}")

target_link_libraries(
    ${LIBNAME}
    PUBLIC
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcupti.so
    PRIVATE
    spdlog::spdlog_header_only
)