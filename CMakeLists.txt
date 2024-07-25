cmake_minimum_required(VERSION 3.16)

# Set the project name and the C++ standard
project(simd_sorting LANGUAGES CXX)

# Set the C++ standard to C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Define the source and header directories
set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)
set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)

# Add the source files
file(GLOB_RECURSE SRC_FILES ${SRC_DIR}/*.cpp)

# Add an executable
add_executable(${PROJECT_NAME} ${SRC_FILES})

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE ${INCLUDE_DIR})

target_compile_options(${PROJECT_NAME} PRIVATE -mmmx -msse -msse2 -msse3 -mssse3 -msse4 -msse4a -msse4.1 -msse4.2 -mavx -mavx2) # -mavx512f -mavx512cd -mavx512vl -mavx512bw -mavx512dq -mavx512vnni)

