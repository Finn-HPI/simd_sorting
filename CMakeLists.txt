cmake_minimum_required(VERSION 3.16)

# Set the project name and the C++ standard
project(simd_sorting LANGUAGES CXX)

# Set the C++ standard to C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Set default L2 Cache size to 256 KiB (262144 bytes)
set(L2_CACHE_SIZE 262144 CACHE STRING "Local L2 Cache Size in bytes")
# Define the L2_CACHE_SIZE macro
add_definitions(-DL2_CACHE_SIZE=${L2_CACHE_SIZE})

# Define the source directories
set(LIB_DIR ${CMAKE_SOURCE_DIR}/lib)
set(BIN_DIR ${CMAKE_SOURCE_DIR}/bin)

# Find Boost (specify components as needed, e.g., system, filesystem, etc.)
find_package(Boost 1.70 REQUIRED COMPONENTS system filesystem)

# Include Boost headers
if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
endif()

# Include directories for the library
add_library(simd_sorting_lib INTERFACE)
target_include_directories(simd_sorting_lib INTERFACE ${LIB_DIR})

# Link Boost libraries to the simd_sorting_lib
target_link_libraries(simd_sorting_lib INTERFACE ${Boost_LIBRARIES})

# Get all .cpp files in the bin directory
file(GLOB_RECURSE BIN_SOURCES ${BIN_DIR}/*.cpp)

# Create executables for each .cpp file found in the bin directory
foreach(SOURCE ${BIN_SOURCES})
    # Get the filename without the directory
    get_filename_component(EXEC_NAME ${SOURCE} NAME_WE)
    add_executable(${EXEC_NAME} ${SOURCE})
    
    # Link each executable to the simd_sorting_lib and Boost libraries
    target_link_libraries(${EXEC_NAME} PRIVATE simd_sorting_lib ${Boost_LIBRARIES})
endforeach()

