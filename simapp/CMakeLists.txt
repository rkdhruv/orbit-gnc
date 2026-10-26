add_executable(simapp main.cpp)
target_link_libraries(simapp PRIVATE simcore rlimgui orbit_warnings)

if(EMSCRIPTEN)
    # Emit simapp.html (+ .js/.wasm). raylib's web backend uses Emscripten's GLFW port.
    set_target_properties(simapp PROPERTIES SUFFIX ".html")
    target_link_options(simapp PRIVATE -sUSE_GLFW=3 -sALLOW_MEMORY_GROWTH=1)
endif()
