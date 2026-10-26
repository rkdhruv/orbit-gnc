include(GoogleTest)

add_executable(simcore_tests
    test_smoke.cpp)
target_link_libraries(simcore_tests PRIVATE simcore GTest::gtest_main orbit_warnings)

gtest_discover_tests(simcore_tests)
