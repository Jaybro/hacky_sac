add_executable(line2 line2.cpp)
set_default_target_properties(line2)
target_link_libraries(line2 PUBLIC hacky_toolkit HackySAC::HackySAC)
