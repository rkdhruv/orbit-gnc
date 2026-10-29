add_library(simcore STATIC
    src/version.cpp
    src/dynamics/point_mass_gravity.cpp
    src/integrators/dense_output.cpp
    src/integrators/dormand_prince.cpp
    src/integrators/fixed_step.cpp
    src/orbits/epoch.cpp)

target_include_directories(simcore PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(simcore
    PUBLIC Eigen3::Eigen
    PRIVATE orbit_warnings)
