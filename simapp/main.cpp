// simapp: raylib window + Dear ImGui panel. Drives simcore only through the Simulation facade:
// it submits commands, steps whole ticks and reads snapshots; it never mutates sim internals.

#include <imgui.h>
#include <raylib.h>
#include <rlImGui.h>

#include <memory>
#include <numbers>

#include "simcore/constants.hpp"
#include "simcore/dynamics/force_sum.hpp"
#include "simcore/dynamics/j2_gravity.hpp"
#include "simcore/dynamics/point_mass_gravity.hpp"
#include "simcore/orbits/elements.hpp"
#include "simcore/sim/simulation.hpp"
#include "simcore/sim/tick_scheduler.hpp"
#include "simcore/version.hpp"

#if defined(__EMSCRIPTEN__)
#include <emscripten/emscripten.h>
#endif

namespace {

constexpr int kWindowWidth = 1280;
constexpr int kWindowHeight = 720;
constexpr double kDeg = std::numbers::pi / 180.0;

// Default scenario: a 500 kg spacecraft with 300 kg of propellant and one 400 N, Isp 320 s main
// engine along body +x, in a 420 km circular orbit at 51.6 deg, under point-mass gravity + J2.
simcore::Simulation make_default_sim() {
    const simcore::VehicleConfig vehicle{
        simcore::MassProperties(
            500.0, Eigen::Vector3d(300.0, 400.0, 450.0).asDiagonal(),
            simcore::PropellantTank{300.0, Eigen::Vector3d(30.0, 30.0, 20.0).asDiagonal()}),
        {simcore::Thruster(400.0, 320.0, Eigen::Vector3d(-1.0, 0.0, 0.0),
                           Eigen::Vector3d::UnitX())},
        {}};
    auto forces = std::make_shared<simcore::ForceSum>();
    forces->add(std::make_shared<simcore::PointMassGravity>())
        .add(std::make_shared<simcore::J2Gravity>());
    const simcore::SimConfig cfg{
        vehicle, forces, std::make_shared<simcore::ConstantBodyTorque>(Eigen::Vector3d::Zero()),
        0.02};

    const simcore::Epoch t0 = simcore::Epoch::from_tt_seconds_since_j2000(830'000'000);
    const simcore::State orbit =
        simcore::state_from_elements(simcore::KeplerianElements{simcore::earth::radius + 420.0e3,
                                                                0.0, 51.6 * kDeg, 0.0, 0.0, 0.0},
                                     t0);
    return simcore::Simulation(cfg, simcore::VehicleState{t0,
                                                          orbit.r_eci,
                                                          orbit.v_eci,
                                                          Eigen::Quaterniond::Identity(),
                                                          Eigen::Vector3d::Zero(),
                                                          300.0,
                                                          {}});
}

struct App {
    simcore::Simulation sim = make_default_sim();
    simcore::TickScheduler scheduler{sim.config().tick_s};
    simcore::Epoch t0 = sim.snapshot().state.t;
    bool fell_behind = false;
};

std::unique_ptr<App> g_app;  // the browser's main-loop callback takes no arguments

// One frame of the app. Shared by the desktop loop and the browser's
// requestAnimationFrame callback (the browser can't run a blocking while-loop).
void UpdateDrawFrame() {
    App& app = *g_app;
    const simcore::TickScheduler::FrameTicks ft =
        app.scheduler.ticks_for_frame(static_cast<double>(GetFrameTime()));
    app.sim.step(ft.ticks);
    app.fell_behind = ft.fell_behind;
    const simcore::Snapshot snap = app.sim.snapshot();

    BeginDrawing();
    ClearBackground(Color{.r = 12, .g = 14, .b = 22, .a = 255});

    rlImGuiBegin();
    ImGui::SetNextWindowPos(ImVec2(16.0F, 16.0F), ImGuiCond_FirstUseEver);
    ImGui::Begin("Orbit GNC");
    ImGui::Text("Mission time  %.2f s", snap.state.t - app.t0);
    ImGui::Text("Tick          %lld", static_cast<long long>(snap.tick));
    ImGui::Text("Altitude      %.3f km",
                (snap.state.r_eci.norm() - simcore::earth::radius) / 1000.0);
    ImGui::Text("Speed         %.3f m/s", snap.state.v_eci.norm());
    ImGui::Text("Propellant    %.1f kg", snap.state.propellant_kg);
    ImGui::Separator();
    ImGui::Text("simcore %.*s  |  %d FPS%s", static_cast<int>(simcore::version().size()),
                simcore::version().data(), GetFPS(), app.fell_behind ? "  (behind)" : "");
    ImGui::End();
    rlImGuiEnd();

    EndDrawing();
}

}  // namespace

// A SIMCORE_CHECK failure escaping main is a programming error, so terminating is intended. M3.6
// shows such errors in the UI instead (exception catching in the web build).
int main() {  // NOLINT(bugprone-exception-escape)
    // raylib's flags are a plain C enum, so OR-ing them is signed-int bitwise by design.
    SetConfigFlags(FLAG_MSAA_4X_HINT | FLAG_VSYNC_HINT |  // NOLINT(bugprone-signed-bitwise)
                   FLAG_WINDOW_RESIZABLE);
    InitWindow(kWindowWidth, kWindowHeight, "Orbit GNC");
    rlImGuiSetup(true);  // true = dark theme
    g_app = std::make_unique<App>();

#if defined(__EMSCRIPTEN__)
    emscripten_set_main_loop(UpdateDrawFrame, 0, 1);
#else
    SetTargetFPS(60);
    while (!WindowShouldClose()) {
        UpdateDrawFrame();
    }
#endif

    g_app.reset();
    rlImGuiShutdown();
    CloseWindow();
    return 0;
}
