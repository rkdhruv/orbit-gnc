add_library(simcore STATIC
    src/version.cpp
    src/dynamics/exponential_drag.cpp
    src/dynamics/force_sum.cpp
    src/dynamics/j2_gravity.cpp
    src/dynamics/point_mass_gravity.cpp
    src/dynamics/rigid_body.cpp
    src/events/brent.cpp
    src/events/events.cpp
    src/integrators/attitude.cpp
    src/integrators/dense_output.cpp
    src/integrators/dormand_prince.cpp
    src/integrators/fixed_step.cpp
    src/integrators/six_dof.cpp
    src/orbits/elements.cpp
    src/orbits/epoch.cpp
    src/orbits/kepler.cpp
    src/vehicle/mass_properties.cpp
    src/vehicle/reaction_wheel.cpp
    src/vehicle/thruster.cpp)

target_include_directories(simcore PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(simcore
    PUBLIC Eigen3::Eigen
    PRIVATE orbit_warnings)
