// simapp: raylib window + Dear ImGui panel. Drives simcore only through the Simulation facade:
// it submits commands, steps whole ticks and reads snapshots; it never mutates sim internals.

#include <imgui.h>
#include <raylib.h>
#include <rlImGui.h>

#include <memory>
#include <numbers>
#include <optional>
#include <string>

#include "controls.hpp"
#include "hud.hpp"
#include "scene.hpp"
#include "simcore/constants.hpp"
#include "simcore/dynamics/force_sum.hpp"
#include "simcore/dynamics/j2_gravity.hpp"
#include "simcore/dynamics/point_mass_gravity.hpp"
#include "simcore/orbits/elements.hpp"
#include "simcore/sim/manual_control.hpp"
#include "simcore/sim/replay.hpp"
#include "simcore/sim/simulation.hpp"
#include "simcore/sim/telemetry.hpp"
#include "simcore/sim/tick_scheduler.hpp"

#if defined(__EMSCRIPTEN__)
#include <emscripten/emscripten.h>
#endif

namespace {

constexpr int kWindowWidth = 1280;
constexpr int kWindowHeight = 720;
constexpr double kDeg = std::numbers::pi / 180.0;

// Default scenario: a 500 kg spacecraft with 300 kg of propellant, one 400 N, Isp 320 s main
// engine along body +x and three reaction wheels on the body axes (their locked inertia is part
// of the dry inertia), in a 420 km circular orbit at 51.6 deg. Two-body gravity allows on-rails
// time warp; with J2 the sim always integrates, so warp stays at or below kMaxIntegratedWarp.
simcore::Simulation make_sim(bool j2) {
    const simcore::VehicleConfig vehicle{
        simcore::MassProperties(
            500.0, Eigen::Vector3d(300.0, 400.0, 450.0).asDiagonal(),
            simcore::PropellantTank{300.0, Eigen::Vector3d(30.0, 30.0, 20.0).asDiagonal()}),
        {simcore::Thruster(400.0, 320.0, Eigen::Vector3d(-1.0, 0.0, 0.0),
                           Eigen::Vector3d::UnitX())},
        {simcore::ReactionWheel(Eigen::Vector3d::UnitX(), 0.05, 0.1, 600.0),
         simcore::ReactionWheel(Eigen::Vector3d::UnitY(), 0.05, 0.1, 600.0),
         simcore::ReactionWheel(Eigen::Vector3d::UnitZ(), 0.05, 0.1, 600.0)}};
    auto forces = std::make_shared<simcore::ForceSum>();
    forces->add(std::make_shared<simcore::PointMassGravity>());
    std::optional<double> rails_mu = simcore::earth::mu;
    if (j2) {
        forces->add(std::make_shared<simcore::J2Gravity>());
        rails_mu.reset();
    }
    const simcore::SimConfig cfg{
        vehicle, forces, std::make_shared<simcore::ConstantBodyTorque>(Eigen::Vector3d::Zero()),
        0.02, rails_mu};

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
                                                          {0.0, 0.0, 0.0}});
}

// The front end's own state; the simulation is only reached through submit / step / snapshot.
struct App {
    explicit App(bool with_j2) : j2(with_j2), sim(make_sim(with_j2)) {}
    bool j2;
    simcore::Simulation sim;
    simcore::VehicleState initial =
        sim.snapshot().state;  // with the input log, replays the session
    std::string replay_status;
    // No practical tick cap: on rails a step() of any length is one Kepler solve, and integrated
    // warp is limited to kMaxIntegratedWarp by the controls instead.
    simcore::TickScheduler scheduler{sim.config().tick_s, 0.25, 1'000'000'000'000};
    simcore::Epoch t0 = sim.snapshot().state.t;
    simapp::Scene scene;
    simapp::Controls controls;
    bool fell_behind = false;
};

std::unique_ptr<App> g_app;  // the browser's main-loop callback takes no arguments

// The first field in which two snapshots differ (bit for bit), or nullptr if none do.
const char* first_difference(const simcore::Snapshot& a, const simcore::Snapshot& b) {
    if (a.tick != b.tick) return "tick";
    if (a.state.t != b.state.t) return "time";
    if (a.state.r_eci != b.state.r_eci) return "position";
    if (a.state.v_eci != b.state.v_eci) return "velocity";
    if (a.state.q_body_to_eci.coeffs() != b.state.q_body_to_eci.coeffs()) return "attitude";
    if (a.state.w_body != b.state.w_body) return "body rates";
    if (a.state.propellant_kg != b.state.propellant_kg) return "propellant";
    if (a.state.wheel_h_nms != b.state.wheel_h_nms) return "wheel momenta";
    if (a.command != b.command) return "command";
    if (a.on_rails != b.on_rails) return "rails mode";
    return nullptr;
}

// Re-run the whole session from its initial state and input log and compare with the live sim.
std::string verify_replay(const App& app) {
    const simcore::Snapshot live = app.sim.snapshot();
    const std::vector<simcore::LoggedCommand> log = app.sim.input_log();
    const simcore::Snapshot again = simcore::replay(app.sim.config(), app.initial, log, live.tick);
    const char* diff = first_difference(live, again);
    return diff == nullptr ? "bit-identical at tick " + std::to_string(live.tick) + " (" +
                                 std::to_string(log.size()) +
                                 (log.size() == 1 ? " logged command)" : " logged commands)")
                           : std::string("MISMATCH: ") + diff + " differs";
}

// One frame of the app. Shared by the desktop loop and the browser's
// requestAnimationFrame callback (the browser can't run a blocking while-loop).
void UpdateDrawFrame() {
    App& app = *g_app;
    const auto frame_dt = static_cast<double>(GetFrameTime());

    // Pilot input -> command for the next ticks. ImGui's capture flags are from the previous
    // frame, which is what input handling needs.
    if (!ImGui::GetIO().WantCaptureKeyboard) {
        app.controls.handle_keys(frame_dt);
    }
    const simcore::VehicleConfig& vehicle = app.sim.config().vehicle;
    const simcore::VehicleCommand cmd =
        simcore::manual_command(vehicle, app.controls.input(), app.sim.snapshot().state);
    app.sim.submit(cmd);
    // Fast warp only on rails: a coast in a two-body environment.
    const bool can_rail = app.sim.config().rails_mu.has_value() && simcore::is_coast(cmd);
    app.controls.clamp_warp(can_rail ? simapp::kWarpLevels.back() : simapp::kMaxIntegratedWarp);

    const simcore::TickScheduler::FrameTicks ft =
        app.scheduler.ticks_for_frame(frame_dt, app.controls.warp());
    app.sim.step(ft.ticks);
    app.fell_behind = ft.fell_behind;
    const simcore::Snapshot snap = app.sim.snapshot();

    app.scene.handle_input(!ImGui::GetIO().WantCaptureMouse);

    BeginDrawing();
    ClearBackground(Color{.r = 4, .g = 5, .b = 10, .a = 255});
    app.scene.draw(snap);

    rlImGuiBegin();
    const simapp::HudAction action = simapp::draw_hud(
        simcore::telemetry(vehicle, snap.state), vehicle,
        simapp::HudFrame{snap.state.t - app.t0, snap.tick, app.scene.camera_distance_m(), GetFPS(),
                         app.fell_behind, app.controls.warp(), app.controls.throttle(),
                         app.controls.rate_damping(), snap.on_rails, app.j2,
                         app.replay_status.c_str()});
    rlImGuiEnd();

    EndDrawing();

    if (action == simapp::HudAction::kVerifyReplay) {
        app.replay_status = verify_replay(app);
    } else if (action != simapp::HudAction::kNone) {
        g_app = std::make_unique<App>(action == simapp::HudAction::kRestartJ2);
    }
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
    g_app = std::make_unique<App>(false);

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
