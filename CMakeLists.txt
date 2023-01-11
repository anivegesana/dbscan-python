cmake_minimum_required(VERSION 3.10)

include(AddGitSubmodule.cmake)

project(dbscan)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CXX_FLAGS "-std=c++17 -pthread -O3 -fPIC")

add_library(dbscan_s src/capi.cpp)
set(DBSCAN_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include")
target_include_directories(dbscan_s PUBLIC ${DBSCAN_INCLUDE_DIR})
target_compile_features(dbscan_s INTERFACE cxx_std_17)

option(DBSCAN_EXECUTABLE  "Build executable"       ON)

if (DBSCAN_EXECUTABLE)
  message(STATUS "executable: Enabled")
  add_subdirectory(executable)
else()
  message(STATUS "executable: Disabled (enable with -DDBSCAN_EXECUTABLE=On)")
endif()

option(DBSCAN_TEST        "Build unit tests"       ON)

if (DBSCAN_TEST)
  message(STATUS "test: Enabled")
  enable_testing()
  add_subdirectory(test)
else()
  message(STATUS "test: Disabled (enable with -DDBSCAN_TEST=On)")
endif()
