# HackySAC

[![build-and-test](https://github.com/Jaybro/hacky_sac/workflows/build-and-test/badge.svg)](https://github.com/Jaybro/hacky_sac/actions?query=workflow%3Abuild-and-test)

HackySAC is a C++ header only library for model estimation using RANSAC.

Available under the [MIT](https://en.wikipedia.org/wiki/MIT_License) license.

# Examples

* [Minimal working example](./examples/line2/line2.cpp) for estimating a line using RANSAC.

# Requirements

Minimum:

* A compiler that is compliant with the C++17 standard or higher.
* [CMake](https://cmake.org/). It is also possible to just copy and paste the [hacky_sac](./src/) directory into an include directory.

Optional:

* [Google Test](https://github.com/google/googletest). Used for running unit tests.
* [Eigen](http://eigen.tuxfamily.org). To run the example and unit tests.

# Build

Build with [CMake](https://cmake.org/):

```console
$ mkdir build && cd build
$ cmake ../
$ cmake --build .
$ cmake --install .
```

```cmake
find_package(HackySAC REQUIRED)

add_executable(myexe main.cpp)
target_link_libraries(myexe PUBLIC HackySAC::HackySAC)
```
