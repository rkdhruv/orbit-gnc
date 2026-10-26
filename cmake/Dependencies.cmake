# Third-party dependencies, all via FetchContent and pinned by URL + SHA256 so a
# given commit of this repo always builds against byte-identical sources.
include(FetchContent)

# --- Eigen (header-only) ----------------------------------------------------
# SOURCE_SUBDIR points at a directory that doesn't exist, so MakeAvailable only
# downloads Eigen and never runs its CMakeLists (which builds tests, docs, BLAS).
FetchContent_Declare(eigen
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.1/eigen-3.4.1.tar.gz
    URL_HASH SHA256=b93c667d1b69265cdb4d9f30ec21f8facbbe8b307cf34c0b9942834c6d4fdbe2
    SOURCE_SUBDIR _no_cmake_)
FetchContent_MakeAvailable(eigen)

add_library(eigen INTERFACE)
add_library(Eigen3::Eigen ALIAS eigen)
target_include_directories(eigen SYSTEM INTERFACE ${eigen_SOURCE_DIR})

# --- GoogleTest ---------------------------------------------------------------
if(ORBIT_BUILD_TESTS)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.tar.gz
        URL_HASH SHA256=65fab701d9829d38cb77c14acdc431d2108bfdbf8979e40eb8ae567edf10b27c)
    FetchContent_MakeAvailable(googletest)
endif()

# --- raylib + Dear ImGui + rlImGui (front end only) ---------------------------
if(ORBIT_BUILD_APP)
    set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(raylib
        URL https://github.com/raysan5/raylib/archive/refs/tags/6.2.tar.gz
        URL_HASH SHA256=df7a13129b1bec9e73a6962efff170beba310ee6eab8f94cd000b7316b4db5ee)

    # Neither ImGui nor rlImGui ships a CMakeLists, so MakeAvailable just
    # downloads them and we define the library targets ourselves below.
    FetchContent_Declare(imgui
        URL https://github.com/ocornut/imgui/archive/refs/tags/v1.93.0.tar.gz
        URL_HASH SHA256=f7bf6bc76e5d79d40d751096663e6c5d18daf08129dd52e7f7b09e4251c8eeee)
    FetchContent_Declare(rlimgui
        URL https://github.com/raylib-extras/rlImGui/archive/1550009359ad975927f7f0e4a3f47e3f27123ea9.tar.gz
        URL_HASH SHA256=7d04ff0c996a89190e1050d3c4448ad784479723c76893efea2d731bd5eb839f)

    FetchContent_MakeAvailable(raylib imgui rlimgui)

    add_library(imgui STATIC
        ${imgui_SOURCE_DIR}/imgui.cpp
        ${imgui_SOURCE_DIR}/imgui_demo.cpp
        ${imgui_SOURCE_DIR}/imgui_draw.cpp
        ${imgui_SOURCE_DIR}/imgui_tables.cpp
        ${imgui_SOURCE_DIR}/imgui_widgets.cpp)
    target_include_directories(imgui SYSTEM PUBLIC ${imgui_SOURCE_DIR})

    add_library(rlimgui STATIC ${rlimgui_SOURCE_DIR}/rlImGui.cpp)
    target_include_directories(rlimgui SYSTEM PUBLIC ${rlimgui_SOURCE_DIR})
    target_link_libraries(rlimgui PUBLIC imgui raylib)
endif()
