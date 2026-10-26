add_library(simcore STATIC
    src/version.cpp)

target_include_directories(simcore PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(simcore
    PUBLIC Eigen3::Eigen
    PRIVATE orbit_warnings)
