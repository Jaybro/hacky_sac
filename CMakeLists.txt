cmake_minimum_required(VERSION 3.12)

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
