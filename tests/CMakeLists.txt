include(GoogleTest)

add_executable(simcore_tests
    test_smoke.cpp
    test_check.cpp
    test_constants.cpp
    orbits/test_epoch.cpp
    orbits/test_elements.cpp
    orbits/test_kepler.cpp
    dynamics/test_state.cpp
    dynamics/test_point_mass_gravity.cpp
    integrators/test_fixed_step.cpp
    events/test_brent.cpp
    events/test_events.cpp
    integrators/test_dormand_prince.cpp
    integrators/test_dense_output.cpp)
target_link_libraries(simcore_tests PRIVATE simcore GTest::gtest_main orbit_warnings)

gtest_discover_tests(simcore_tests)
