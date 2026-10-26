// simapp: raylib window + Dear ImGui panel. Reads simcore; never mutates it.

#include <imgui.h>
#include <raylib.h>
#include <rlImGui.h>

#include "simcore/version.hpp"

#if defined(__EMSCRIPTEN__)
#include <emscripten/emscripten.h>
#endif

namespace {

constexpr int kWindowWidth = 1280;
constexpr int kWindowHeight = 720;

// One frame of the app. Shared by the desktop loop and the browser's
// requestAnimationFrame callback (the browser can't run a blocking while-loop).
void UpdateDrawFrame() {
    BeginDrawing();
    ClearBackground(Color{.r = 12, .g = 14, .b = 22, .a = 255});

    rlImGuiBegin();
    ImGui::SetNextWindowPos(ImVec2(16.0F, 16.0F), ImGuiCond_FirstUseEver);
    ImGui::Begin("Orbit GNC");
    ImGui::Text("Hello, orbit");
    ImGui::Separator();
    ImGui::Text("simcore %.*s", static_cast<int>(simcore::version().size()),
                simcore::version().data());
    ImGui::Text("%d FPS", GetFPS());
    ImGui::End();
    rlImGuiEnd();

    EndDrawing();
}

}  // namespace

int main() {
    // raylib's flags are a plain C enum, so OR-ing them is signed-int bitwise by design.
    SetConfigFlags(FLAG_MSAA_4X_HINT | FLAG_VSYNC_HINT |  // NOLINT(bugprone-signed-bitwise)
                   FLAG_WINDOW_RESIZABLE);
    InitWindow(kWindowWidth, kWindowHeight, "Orbit GNC");
    rlImGuiSetup(true);  // true = dark theme

#if defined(__EMSCRIPTEN__)
    emscripten_set_main_loop(UpdateDrawFrame, 0, 1);
#else
    SetTargetFPS(60);
    while (!WindowShouldClose()) {
        UpdateDrawFrame();
    }
#endif

    rlImGuiShutdown();
    CloseWindow();
    return 0;
}
