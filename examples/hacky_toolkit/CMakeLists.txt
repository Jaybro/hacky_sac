add_library(hacky_toolkit INTERFACE)
target_include_directories(hacky_toolkit INTERFACE ${CMAKE_CURRENT_LIST_DIR})
target_compile_features(hacky_toolkit INTERFACE cxx_std_17)
target_link_libraries(hacky_toolkit INTERFACE Eigen3::Eigen)
target_sources(hacky_toolkit
    INTERFACE
        ${CMAKE_CURRENT_LIST_DIR}/hacky_toolkit/line2.hpp
        ${CMAKE_CURRENT_LIST_DIR}/hacky_toolkit/plane.hpp
)
