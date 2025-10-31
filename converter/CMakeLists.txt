cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(NPSConverter LANGUAGES CXX)

# --- Find required packages ---
find_package(ROOT REQUIRED COMPONENTS Core RIO Tree)
find_package(Torch REQUIRED)

# --- Include ROOT headers ---
include_directories(${ROOT_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include)
add_definitions(${ROOT_DEFINITIONS})

# --- Handle C++ flags ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Suppress "glog" conflicts with LibTorch/ROOT macros
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)

# --- Common source files (prepend with sources/) ---
file(GLOB PROJECT_SOURCES ${CMAKE_SOURCE_DIR}/sources/*.cpp)

add_library(NPSConverter ${PROJECT_SOURCES})

# --- Link required libraries for the library target ---
target_link_libraries(
    NPSConverter
    PUBLIC
    ${ROOT_LIBRARIES}
    ${TORCH_LIBRARIES}
)

# --- Include headers for library consumers (e.g., standalone executables) ---
target_include_directories(
    NPSConverter
    PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${ROOT_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
)

add_subdirectory(standalone)
