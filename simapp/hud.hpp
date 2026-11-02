#pragma once

#include <cstdint>

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
};

// The ImGui panel: Orbit, Vehicle, Attitude and Wheels sections. Only formats telemetry (simcore
// computes it); SI is converted to km, minutes and degrees here, at the UI.
void draw_hud(const simcore::Telemetry& tm, const simcore::VehicleConfig& cfg, const HudFrame& f);

}  // namespace simapp
