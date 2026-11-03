#pragma once

#include <cstdint>
#include <optional>

#include "simcore/gnc/attitude_control.hpp"
#include "simcore/sim/telemetry.hpp"
#include "simcore/vehicle/vehicle.hpp"

namespace simapp {

// Per-frame values the HUD shows that aren't vehicle telemetry.
struct HudFrame {
    double mission_time_s;
    std::int64_t tick;
    double camera_distance_m;
    int fps;
    bool fell_behind;
    // Flight controls and mode.
    double warp;
    double throttle;
    bool rate_damping;
    std::optional<simcore::HoldMode> hold;  // active attitude hold, if any
    bool use_lqr;                           // hold controller: LQR (true) or PD
    bool dumping;                           // wheel momentum dump (RCS) in progress
    bool on_rails;
    bool j2;                    // perturbed environment: J2 + gravity gradient (no rails warp)
    const char* replay_status;  // result of the last "Verify replay", or "" if none yet
};

// The maneuver-node editor's inputs (front-end state, edited by the HUD). Prograde is along-track
// (S), normal along h (W), radial along r (R).
struct NodeEditor {
    double prograde_mps{50.0};
    double normal_mps{0.0};
    double radial_mps{0.0};
    double in_s{600.0};  // node time from now
    double hohmann_alt_km{35786.0};
    double target_inc_deg{0.0};
};

// What the HUD shows about the planned node (computed by the app each frame).
struct NodeView {
    bool has_node{false};
    double t_to_node_s{0.0};
    double dv_mps{0.0};
    double burn_s{0.0};
    double t_to_ignition_s{0.0};
    bool predicted_valid{false};  // the post-burn orbit has conic elements
    double peri_alt_m{0.0};
    double apo_alt_m{0.0};
    bool executing{false};
    const char* phase{""};
    double achieved_mps{0.0};
    int queued{0};           // further planned steps after this node
    const char* status{""};  // last planning message (e.g. why a calculator refused)
};

// What the pilot clicked this frame.
enum class HudAction : std::uint8_t {
    kNone,
    kRestartTwoBody,
    kRestartJ2,   // the perturbed scenario: J2 + gravity gradient
    kRestartGeo,  // the GEO mission (upper stage at 300 km equatorial)
    kVerifyReplay,
    kToggleController,
    kAddNode,
    kAddNodeAtApoapsis,
    kAddNodeAtPeriapsis,
    kHohmann,
    kCircularizeApoapsis,
    kCircularizePeriapsis,
    kPlaneChange,
    kDeleteNode,  // clears the whole plan
    kExecuteNode,
    kCancelExecute,
    kHoldOff,
    kHoldPrograde,  // the kHold* values after kHoldOff follow HoldMode's order
    kHoldRetrograde,
    kHoldNormal,
    kHoldAntiNormal,
    kHoldRadialOut,
    kHoldRadialIn,
};

// The ImGui panel: Flight, Maneuver, Orbit, Vehicle, Attitude and Wheels sections. Only formats
// telemetry (simcore computes it); SI is converted to km, minutes and degrees here, at the UI.
HudAction draw_hud(const simcore::Telemetry& tm, const simcore::VehicleConfig& cfg,
                   const HudFrame& f, NodeEditor& editor, const NodeView& node);

}  // namespace simapp
