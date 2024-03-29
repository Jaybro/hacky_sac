cmake_minimum_required(VERSION 3.12)

include (${CMAKE_CURRENT_SOURCE_DIR}/cmake/utils.cmake)

project(hacky_sac
    LANGUAGES CXX
    VERSION 0.1.0
    DESCRIPTION "HackySAC is a C++ header only library for model estimation using RANSAC."
    HOMEPAGE_URL "https://github.com/Jaybro/hacky_sac")

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

set(PROJECT_PACKAGE_NAME "HackySAC")
add_subdirectory(src)

option(BUILD_EXAMPLES "Enable the creation of examples." ON)
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

include(CTest)
find_package(GTest QUIET)
find_package(Eigen3 QUIET)
if(BUILD_TESTING)
    if (GTEST_FOUND AND Eigen3_FOUND)
        # Tests are dependent on some common code.
        # For now, the understory is considered important enough to be tested.
        if(NOT TARGET hacky_toolkit)
            add_subdirectory(examples/hacky_toolkit)
        endif()

        enable_testing()
        add_subdirectory(test)
        message(STATUS "GTest and Eigen found. Building unit tests.")
    else()
        message(STATUS "GTest and/or Eigen not found. Unit tests will not be build.")
    endif()
endif()
