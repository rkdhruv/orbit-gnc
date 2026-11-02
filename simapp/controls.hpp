#pragma once

#include <array>
#include <cstddef>

#include <Eigen/Core>

#include "simcore/sim/manual_control.hpp"

namespace simapp {

// Time-warp levels (sim seconds per real second). Above kMaxIntegratedWarp the sim must be on
// rails: integrating 50 Hz ticks any faster costs too much per frame.
inline constexpr std::array<double, 9> kWarpLevels{1.0,   2.0,    5.0,   10.0, 50.0,
                                                   100.0, 1000.0, 1.0e4, 1.0e5};
inline constexpr double kMaxIntegratedWarp = 100.0;

// Keyboard state the pilot builds up, KSP-style:
//   Shift / Ctrl  throttle up / down (50% per second)    Z / X  full throttle / cut
//   W / S  pitch (nose down / up)   A / D  yaw left / right   Q / E  roll left / right
//   T  rate damping on / off        , / .  warp down / up
// Body axes: x forward (the engine thrusts along +x), y left, z up.
class Controls {
public:
    void handle_keys(double frame_dt_s);  // reads raylib's keyboard
    [[nodiscard]] simcore::ManualInput input() const {
        return simcore::ManualInput{throttle_, rotate_, rate_damping_};
    }
    [[nodiscard]] double warp() const { return kWarpLevels.at(warp_index_); }
    // Lower the warp to at most max_warp (when the vehicle can't go on rails).
    void clamp_warp(double max_warp);
    [[nodiscard]] double throttle() const { return throttle_; }
    [[nodiscard]] bool rate_damping() const { return rate_damping_; }

private:
    double throttle_{0.0};
    Eigen::Vector3d rotate_{Eigen::Vector3d::Zero()};
    bool rate_damping_{true};
    std::size_t warp_index_{0};
};

}  // namespace simapp
