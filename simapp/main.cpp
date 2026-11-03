// simapp: raylib window + Dear ImGui panel. Drives simcore only through the Simulation facade:
// it submits commands, steps whole ticks and reads snapshots; it never mutates sim internals.

#include <imgui.h>
#include <raylib.h>
#include <rlImGui.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <exception>
#include <limits>
#include <memory>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <string>

#include "controls.hpp"
#include "hud.hpp"
#include "scene.hpp"
#include "simcore/constants.hpp"
#include "simcore/dynamics/force_sum.hpp"
#include "simcore/dynamics/gravity_gradient.hpp"
#include "simcore/dynamics/j2_gravity.hpp"
#include "simcore/dynamics/point_mass_gravity.hpp"
#include "simcore/gnc/attitude_control.hpp"
#include "simcore/gnc/maneuver.hpp"
#include "simcore/gnc/momentum.hpp"
#include "simcore/gnc/transfers.hpp"
#include "simcore/gnc/wheel_allocation.hpp"
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
// engine along body +x, a product of inertia I_xz = 150 kg m^2 (asymmetric appendages: without
// one every hold frame puts a principal axis along r and gravity gradient never loads the wheels;
// in a prograde hold it gives 3 n^2 I_xz = 5.7e-4 N m, a dump every ~12 h), three reaction wheels
// on the body axes (J = 0.05, 30 N m s each, room for a 90 deg slew's ~8 N m s; their locked
// inertia is part of the dry inertia) and RCS couples about each axis (0.5 N m, 1 m lever, Isp 220
// s, fed from the main tank: an approximation, real hydrazine RCS would have its own), in a 420 km
// circular orbit at 51.6 deg. Two-body gravity allows on-rails time warp. The perturbed scenario
// adds J2 and gravity-gradient torque (inertia fixed at the full-tank value, a stated
// simplification); it always integrates, so warp stays at or below kMaxIntegratedWarp.
simcore::Simulation make_sim(bool perturbed) {
    const simcore::VehicleConfig vehicle{
        simcore::MassProperties(
            500.0,
            (Eigen::Matrix3d() << 300.0, 0.0, 150.0, 0.0, 400.0, 0.0, 150.0, 0.0, 450.0).finished(),
            simcore::PropellantTank{300.0, Eigen::Vector3d(30.0, 30.0, 20.0).asDiagonal()}),
        {simcore::Thruster(400.0, 320.0, Eigen::Vector3d(-1.0, 0.0, 0.0),
                           Eigen::Vector3d::UnitX())},
        {simcore::ReactionWheel(Eigen::Vector3d::UnitX(), 0.05, 0.1, 600.0),
         simcore::ReactionWheel(Eigen::Vector3d::UnitY(), 0.05, 0.1, 600.0),
         simcore::ReactionWheel(Eigen::Vector3d::UnitZ(), 0.05, 0.1, 600.0)},
        {simcore::RcsAxis(Eigen::Vector3d::UnitX(), 0.5, 1.0, 220.0),
         simcore::RcsAxis(Eigen::Vector3d::UnitY(), 0.5, 1.0, 220.0),
         simcore::RcsAxis(Eigen::Vector3d::UnitZ(), 0.5, 1.0, 220.0)}};
    auto forces = std::make_shared<simcore::ForceSum>();
    forces->add(std::make_shared<simcore::PointMassGravity>());
    std::optional<double> rails_mu = simcore::earth::mu;
    std::shared_ptr<const simcore::TorqueModel> torques =
        std::make_shared<simcore::ConstantBodyTorque>(Eigen::Vector3d::Zero());
    if (perturbed) {
        forces->add(std::make_shared<simcore::J2Gravity>());
        torques =
            std::make_shared<simcore::GravityGradientTorque>(vehicle.mass.inertia_kgm2(300.0));
        rails_mu.reset();
    }
    const simcore::SimConfig cfg{vehicle, forces, torques, 0.02, rails_mu};

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
    std::string error;  // set when the sim throws; the app then shows it and stops stepping
    // No practical tick cap: on rails a step() of any length is one Kepler solve, and integrated
    // warp is limited to kMaxIntegratedWarp by the controls instead.
    simcore::TickScheduler scheduler{sim.config().tick_s, 0.25, 1'000'000'000'000};
    simcore::Epoch t0 = sim.snapshot().state.t;
    simapp::Scene scene;
    simapp::Controls controls;
    bool fell_behind = false;
    bool dumping = false;  // wheel momentum dump in progress (hysteresis state)
    simapp::NodeEditor node_editor;
    std::optional<simcore::ManeuverNode> node;      // the next planned burn, if any
    std::optional<simcore::ManeuverExecutor> exec;  // flying it, if Execute was pressed
    // Steps after `node`. A circularise step is computed only when its turn comes, from the orbit
    // the previous burns actually produced (so a finite-burn error isn't carried forward).
    enum class StepKind : std::uint8_t { kNode, kCircularizeApoapsis, kCircularizePeriapsis };
    struct PlanStep {
        StepKind kind;
        simcore::ManeuverNode node;  // for kNode
    };
    std::deque<PlanStep> queue;
    bool auto_continue = false;  // Execute flies the whole plan
    std::string plan_status;
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

// Start flying the current node (if any), reporting a refusal in the plan status.
void start_executor(App& app, const simcore::VehicleState& s) {
    if (!app.node) {
        return;
    }
    try {
        app.exec.emplace(app.sim.config().vehicle, s, *app.node, simcore::kManualHoldMax_s);
    } catch (const std::exception& e) {
        app.plan_status = std::string("can't execute: ") + e.what();
        app.auto_continue = false;
    }
}

// Append a step to the plan: it becomes the current node if there is none.
void push_step(App& app, const App::PlanStep& step, const simcore::VehicleState& s) {
    if (app.node || !app.queue.empty()) {
        app.queue.push_back(step);
        return;
    }
    const simcore::State orbit{s.t, s.r_eci, s.v_eci};
    switch (step.kind) {
        case App::StepKind::kNode:
            app.node = step.node;
            break;
        case App::StepKind::kCircularizeApoapsis:
            app.node = simcore::circularize_node(orbit, true);
            break;
        case App::StepKind::kCircularizePeriapsis:
            app.node = simcore::circularize_node(orbit, false);
            break;
    }
}

// Planning actions and the hand-on from one finished node to the next. Calculator refusals
// (e.g. a plane change on an equatorial orbit) land in the plan status instead of stopping the sim.
void plan_frame(App& app, const simcore::VehicleState& s, simapp::HudAction action) {
    using simapp::HudAction;
    const simcore::State orbit{s.t, s.r_eci, s.v_eci};
    const simapp::NodeEditor& e = app.node_editor;
    try {
        if (app.exec && app.exec->phase() == simcore::ManeuverExecutor::Phase::kDone) {
            app.exec.reset();
            app.node.reset();
            if (!app.queue.empty()) {
                const App::PlanStep next = app.queue.front();
                app.queue.pop_front();
                push_step(app, next, s);  // circularise steps are computed now, from the real orbit
                if (app.auto_continue) {
                    start_executor(app, s);
                }
            } else {
                app.auto_continue = false;
                app.plan_status = "plan complete";
            }
        }
        const Eigen::Vector3d dv_rsw(e.radial_mps, e.prograde_mps, e.normal_mps);  // RSW order
        if (action == HudAction::kAddNode && std::isfinite(e.in_s) && e.in_s >= 0.0 &&
            dv_rsw.norm() > 0.0) {
            push_step(app, {App::StepKind::kNode, {s.t + e.in_s, dv_rsw}}, s);
        } else if ((action == HudAction::kAddNodeAtApoapsis ||
                    action == HudAction::kAddNodeAtPeriapsis) &&
                   dv_rsw.norm() > 0.0) {
            const double nu = action == HudAction::kAddNodeAtApoapsis ? std::numbers::pi : 0.0;
            push_step(
                app,
                {App::StepKind::kNode, {s.t + simcore::time_to_true_anomaly_s(orbit, nu), dv_rsw}},
                s);
        } else if (action == HudAction::kHohmann) {
            const double r2 = simcore::earth::radius + 1000.0 * e.hohmann_alt_km;
            // First burn: on a near-circular orbit, 300 s (time to slew) plus the burn's own lead
            // from now; otherwise at periapsis when raising and apoapsis when lowering, where the
            // velocity is horizontal.
            const simcore::KeplerianElements el = simcore::elements_from_state(orbit);
            simcore::Epoch t_burn = s.t + 300.0;
            if (el.e >= 1e-3) {
                const bool raising = r2 > el.a_m;
                t_burn =
                    s.t + simcore::time_to_true_anomaly_s(orbit, raising ? 0.0 : std::numbers::pi);
            } else if (!app.sim.config().vehicle.thrusters.empty()) {
                const simcore::VehicleConfig& v = app.sim.config().vehicle;
                const double dv1 = std::abs(simcore::hohmann_transfer(orbit, t_burn, r2).dv1_mps);
                t_burn += simcore::burn_lead_s(dv1, v.mass.mass_kg(s.propellant_kg),
                                               v.thrusters.front().force_body_n(1.0).norm(),
                                               v.thrusters.front().exhaust_velocity_mps());
            }
            const simcore::HohmannPlan p = simcore::hohmann_transfer(orbit, t_burn, r2);
            push_step(app, {App::StepKind::kNode, p.first}, s);
            push_step(app,
                      {p.dv1_mps > 0.0 ? App::StepKind::kCircularizeApoapsis
                                       : App::StepKind::kCircularizePeriapsis,
                       {}},
                      s);
            std::array<char, 128> msg{};
            std::snprintf(msg.data(), msg.size(),
                          "Hohmann: %.1f + %.1f = %.1f m/s, %.2f h transfer", p.dv1_mps, p.dv2_mps,
                          std::abs(p.dv1_mps) + std::abs(p.dv2_mps), p.transfer_s / 3600.0);
            app.plan_status = msg.data();
        } else if (action == HudAction::kCircularizeApoapsis) {
            push_step(app, {App::StepKind::kCircularizeApoapsis, {}}, s);
        } else if (action == HudAction::kCircularizePeriapsis) {
            push_step(app, {App::StepKind::kCircularizePeriapsis, {}}, s);
        } else if (action == HudAction::kPlaneChange) {
            push_step(
                app,
                {App::StepKind::kNode,
                 simcore::plane_change_node(orbit, e.target_inc_deg * std::numbers::pi / 180.0)},
                s);
        } else if (action == HudAction::kDeleteNode) {
            app.node.reset();
            app.queue.clear();
            app.plan_status.clear();
        } else if (action == HudAction::kExecuteNode && app.node) {
            app.auto_continue = true;
            app.plan_status.clear();
            start_executor(app, s);
        } else if (action == HudAction::kCancelExecute) {
            app.exec.reset();
            app.auto_continue = false;
        }
    } catch (const std::invalid_argument& ex) {
        app.plan_status = std::string("can't plan that: ") + ex.what();
    }
}

// The sim half of a frame: pilot input -> command, warp limit, ticks. May throw (a SIMCORE_CHECK
// failure, e.g. rails on a near-parabolic orbit); it runs before any drawing starts, so an
// exception never leaves raylib or ImGui mid-frame.
simcore::Telemetry update_sim(App& app, double frame_dt) {
    // ImGui's capture flags are from the previous frame, which is what input handling needs.
    if (!ImGui::GetIO().WantCaptureKeyboard) {
        app.controls.handle_keys(frame_dt);
    }
    const simcore::VehicleConfig& vehicle = app.sim.config().vehicle;
    const simcore::ManualInput input = app.controls.input();
    // The control law for this frame: an attitude hold if one is on (and the rotate keys are
    // idle and there is an orbit frame), otherwise manual flight with optional rate damping.
    const std::optional<simcore::HoldMode> hold = app.controls.hold();
    const bool use_lqr = app.controls.use_lqr();
    // While a hold is active, wheel momentum is managed too: dumping (with hysteresis) whenever the
    // wheels near saturation. The hysteresis flag lives in App; every resulting command is
    // submitted and logged, so replay is unaffected.
    bool& dumping = app.dumping;
    // Attitude gains from the current inertia (it changes as propellant burns), PD or LQR.
    const auto gains_for = [&vehicle, use_lqr](const simcore::VehicleState& s) {
        const Eigen::Matrix3d I = vehicle.mass.inertia_kgm2(s.propellant_kg);
        double tau_max = std::numeric_limits<double>::infinity();
        for (int k = 0; k < 3; ++k) {
            tau_max = std::min(tau_max, simcore::wheel_axis_authority_nm(vehicle, k));
        }
        return use_lqr && tau_max > 0.0
                   ? simcore::lqr_attitude_gains_bryson(
                         I, simcore::BrysonLimits{simapp::kLqrMaxAngle_rad,
                                                  simapp::kLqrMaxRate_radps, tau_max})
                   : simcore::pd_gains(I, simapp::kHoldGains);
    };
    // A node being executed takes over completely (pilot inputs are ignored until it is done or
    // cancelled); the executor's phase lives in App, and its commands are logged like any other.
    std::optional<simcore::ManeuverExecutor>& exec = app.exec;
    const simcore::ControlLaw law = [&vehicle, input, hold, &dumping, &exec,
                                     &gains_for](const simcore::VehicleState& s) {
        if (exec && exec->phase() != simcore::ManeuverExecutor::Phase::kDone) {
            dumping = false;
            return exec->command(vehicle, s, gains_for(s));
        }
        const bool has_frame =
            s.r_eci.cross(s.v_eci).norm() > 1e-12 * s.r_eci.norm() * s.v_eci.norm();
        if (hold && input.rotate == Eigen::Vector3d::Zero() && has_frame) {
            const simcore::AttitudeGains gains = gains_for(s);
            dumping = simcore::desat_active(dumping, vehicle, s, simcore::DesatSettings{});
            return simcore::attitude_command_with_desat(
                vehicle, s, simcore::hold_target(*hold, simcore::State{s.t, s.r_eci, s.v_eci}),
                gains, dumping, simcore::DesatSettings{}, input.throttle);
        }
        dumping = false;
        return simcore::manual_command(vehicle, input, s);
    };
    // Fast warp only if the next tick would be a rails coast (coasting, not rotating, a conic
    // orbit clear of the surface, two-body environment); otherwise integration's limit.
    const bool can_rail = app.sim.would_rail(law(app.sim.snapshot().state));
    app.controls.clamp_warp(can_rail ? simapp::kWarpLevels.back() : simapp::kMaxIntegratedWarp);

    const simcore::TickScheduler::FrameTicks ft =
        app.scheduler.ticks_for_frame(frame_dt, app.controls.warp());
    std::int64_t ticks = ft.ticks;
    app.fell_behind = ft.fell_behind;
    if (!can_rail && ticks > simapp::kMaxIntegratedTicksPerFrame) {
        ticks = simapp::kMaxIntegratedTicksPerFrame;
        app.fell_behind = true;
    }
    // Recomputes the command at least every kManualHoldMax_s, so damping and holds stay stable at
    // any frame rate and warp.
    simcore::fly_with(app.sim, law, ticks);
    if (app.sim.snapshot().impacted) {
        throw std::runtime_error("impact: the vehicle hit the surface (altitude below 0 km)");
    }
    return simcore::telemetry(vehicle, app.sim.snapshot().state);
}

// Shown instead of the HUD once the sim has thrown: the message and a way to start again.
simapp::HudAction draw_error(const std::string& message, bool j2) {
    ImGui::SetNextWindowPos(ImVec2(16.0F, 16.0F), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(420.0F, 0.0F), ImGuiCond_Always);
    ImGui::Begin("Simulation stopped");
    ImGui::TextWrapped("The simulation rejected its state and stopped:");
    ImGui::TextWrapped("%s", message.c_str());
    simapp::HudAction action = simapp::HudAction::kNone;
    if (ImGui::Button("Restart")) {
        action = j2 ? simapp::HudAction::kRestartJ2 : simapp::HudAction::kRestartTwoBody;
    }
    ImGui::End();
    return action;
}

// One frame of the app. Shared by the desktop loop and the browser's
// requestAnimationFrame callback (the browser can't run a blocking while-loop).
void UpdateDrawFrame() {
    App& app = *g_app;
    std::optional<simcore::Telemetry> tm;
    if (app.error.empty()) {
        try {
            tm = update_sim(app, static_cast<double>(GetFrameTime()));
        } catch (const std::exception& e) {
            app.error = e.what();  // the sim stops here; nothing steps until a restart
        }
    }
    const simcore::Snapshot snap = app.sim.snapshot();
    const simcore::VehicleConfig& vehicle = app.sim.config().vehicle;

    app.scene.handle_input(!ImGui::GetIO().WantCaptureMouse);

    BeginDrawing();
    ClearBackground(Color{.r = 4, .g = 5, .b = 10, .a = 255});
    // The planned node: its predicted (impulsive, two-body) result, for the scene and the HUD.
    std::optional<simcore::State> predicted;
    simapp::NodeView node_view;
    if (app.node && app.error.empty()) {
        const simcore::State now{snap.state.t, snap.state.r_eci, snap.state.v_eci};
        try {
            predicted = simcore::post_burn_state(now, *app.node);
        } catch (const std::invalid_argument&) {
            predicted.reset();  // a non-conic orbit has no Kepler prediction
        }
        node_view.has_node = true;
        node_view.t_to_node_s = app.node->t - snap.state.t;
        node_view.dv_mps = app.node->dv_rsw_mps.norm();
        if (!vehicle.thrusters.empty()) {
            const double m = vehicle.mass.mass_kg(snap.state.propellant_kg);
            const double thrust = vehicle.thrusters.front().force_body_n(1.0).norm();
            const double ve = vehicle.thrusters.front().exhaust_velocity_mps();
            node_view.burn_s = simcore::burn_duration_s(node_view.dv_mps, m, thrust, ve);
            node_view.t_to_ignition_s =
                node_view.t_to_node_s - simcore::burn_lead_s(node_view.dv_mps, m, thrust, ve);
        }
        if (predicted && simcore::has_conic_elements(*predicted)) {
            const simcore::KeplerianElements el = simcore::elements_from_state(*predicted);
            node_view.predicted_valid = true;
            node_view.peri_alt_m = el.a_m * (1.0 - el.e) - simcore::earth::radius;
            node_view.apo_alt_m = el.e < 1.0 ? el.a_m * (1.0 + el.e) - simcore::earth::radius
                                             : std::numeric_limits<double>::infinity();
        }
        node_view.queued = static_cast<int>(app.queue.size());
        if (app.exec) {
            using Phase = simcore::ManeuverExecutor::Phase;
            node_view.executing = true;
            node_view.phase =
                app.exec->phase() == Phase::kAlign
                    ? (app.exec->ignition_delayed() ? "aligning (ignition held until aligned)"
                                                    : "aligning, waiting")
                : app.exec->phase() == Phase::kBurn ? "burning"
                                                    : "done";
            node_view.achieved_mps = app.exec->achieved_dv_mps();
        }
    }
    node_view.status = app.plan_status.c_str();
    app.scene.draw(snap, predicted);

    rlImGuiBegin();
    const simapp::HudAction action =
        tm.has_value()
            ? simapp::draw_hud(
                  *tm, vehicle,
                  simapp::HudFrame{snap.state.t - app.t0, snap.tick, app.scene.camera_distance_m(),
                                   GetFPS(), app.fell_behind, app.controls.warp(),
                                   app.controls.throttle(), app.controls.rate_damping(),
                                   app.controls.hold(), app.controls.use_lqr(), app.dumping,
                                   snap.on_rails, app.j2, app.replay_status.c_str()},
                  app.node_editor, node_view)
            : draw_error(app.error, app.j2);
    rlImGuiEnd();

    EndDrawing();

    plan_frame(app, snap.state, action);
    if (action == simapp::HudAction::kToggleController) {
        app.controls.toggle_lqr();
    } else if (action >= simapp::HudAction::kHoldOff) {
        const auto i = static_cast<int>(action) - static_cast<int>(simapp::HudAction::kHoldOff);
        app.controls.set_hold(
            i == 0 ? std::nullopt
                   : std::optional<simcore::HoldMode>(static_cast<simcore::HoldMode>(i - 1)));
    } else if (action == simapp::HudAction::kVerifyReplay) {
        try {
            app.replay_status = verify_replay(app);
        } catch (const std::exception& e) {
            app.replay_status = std::string("replay failed: ") + e.what();
        }
    } else if (action == simapp::HudAction::kRestartTwoBody ||
               action == simapp::HudAction::kRestartJ2) {
        g_app = std::make_unique<App>(action == simapp::HudAction::kRestartJ2);
    }
}

}  // namespace

// Sim errors are caught per frame and shown in the UI (the web build enables exception catching
// for this). What can still escape main is a failure building the initial scenario, a programming
// error, for which terminating is the right outcome.
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
