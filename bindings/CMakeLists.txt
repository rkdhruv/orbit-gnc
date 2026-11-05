# The Python module `orbitgnc`: pybind11 wrappers over simcore, no logic of its own.
pybind11_add_module(orbitgnc orbitgnc.cpp)
target_link_libraries(orbitgnc PRIVATE simcore orbit_warnings)
# simcore is a static library linked into a shared module, so it must be position-independent.
set_target_properties(simcore PROPERTIES POSITION_INDEPENDENT_CODE ON)
# One place to find the module: build/<preset>/python/orbitgnc.*.so (PYTHONPATH for tests and
# notebooks).
set_target_properties(orbitgnc PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python)
