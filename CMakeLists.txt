cmake_minimum_required(VERSION 3.25)

project(orbit_gnc VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ORBIT_BUILD_APP "Build the raylib/ImGui front end (simapp)" ON)
option(ORBIT_BUILD_TESTS "Build the GoogleTest suite (simcore_tests)" ON)
option(ORBIT_WERROR "Treat warnings in our own targets as errors" OFF)

# Determinism: forbid the compiler from fusing a*b + c into a single FMA.
# An FMA rounds once instead of twice, so whether it's used changes the last
# bit of results depending on compiler/CPU. Set before any dependency is added
# so it applies to every target in the build, third-party ones included.
add_compile_options(-ffp-contract=off)

# Warning set for our own code only (dependencies are not held to it).
add_library(orbit_warnings INTERFACE)
target_compile_options(orbit_warnings INTERFACE
    -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion
    $<$<BOOL:${ORBIT_WERROR}>:-Werror>)

include(cmake/Dependencies.cmake)

add_subdirectory(simcore)

if(ORBIT_BUILD_APP)
    add_subdirectory(simapp)
endif()

if(ORBIT_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
