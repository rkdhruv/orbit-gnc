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
    bool on_rails;
    bool j2;                    // environment includes J2 (no rails warp)
    const char* replay_status;  // result of the last "Verify replay", or "" if none yet
};

// What the pilot clicked this frame.
enum class HudAction : std::uint8_t {
    kNone,
    kRestartTwoBody,
    kRestartJ2,
    kVerifyReplay,
    kHoldOff,
    kHoldPrograde,  // the kHold* values after kHoldOff follow HoldMode's order
    kHoldRetrograde,
    kHoldNormal,
    kHoldAntiNormal,
    kHoldRadialOut,
    kHoldRadialIn,
};

// The ImGui panel: Flight, Orbit, Vehicle, Attitude and Wheels sections. Only formats telemetry
// (simcore computes it); SI is converted to km, minutes and degrees here, at the UI.
HudAction draw_hud(const simcore::Telemetry& tm, const simcore::VehicleConfig& cfg,
                   const HudFrame& f);

}  // namespace simapp
