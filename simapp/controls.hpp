#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

#include <Eigen/Core>

#include "simcore/gnc/attitude_control.hpp"
#include "simcore/sim/manual_control.hpp"

namespace simapp {

// Time-warp levels (sim seconds per real second). Above kMaxIntegratedWarp the sim must be on
// rails: integrating 50 Hz ticks any faster costs too much per frame.
inline constexpr std::array<double, 9> kWarpLevels{1.0,   2.0,    5.0,   10.0, 50.0,
                                                   100.0, 1000.0, 1.0e4, 1.0e5};
inline constexpr double kMaxIntegratedWarp = 100.0;
// Backstop: at most this many integrated (RK4) ticks per frame, whatever the warp (~5 ms natively
// at ~1 us per tick). Time beyond it is dropped and the HUD shows "behind".
inline constexpr std::int64_t kMaxIntegratedTicksPerFrame = 5000;
// Attitude-hold gains: wn = 0.05 rad/s (a ~2 min settling time), zeta = 0.7. A small correction
// stays within the default vehicle's 0.1 N m wheels; a large slew saturates them, and the wheel
// limit then sets how fast it turns.
inline constexpr simcore::PdGains kHoldGains{0.05, 0.7};

// Keyboard state the pilot builds up, KSP-style:
//   Shift / Ctrl  throttle up / down (50% per second)    Z / X  full throttle / cut
//   W / S  pitch (nose down / up)   A / D  yaw left / right   Q / E  roll left / right
//   T  rate damping on / off        R  cycle attitude hold (off, prograde, ..., radial in)
//   , / .  warp down / up
// An attitude hold, when on, replaces rate damping; the rotate keys override it while held.
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
    [[nodiscard]] std::optional<simcore::HoldMode> hold() const { return hold_; }
    void set_hold(std::optional<simcore::HoldMode> mode) { hold_ = mode; }

private:
    double throttle_{0.0};
    Eigen::Vector3d rotate_{Eigen::Vector3d::Zero()};
    bool rate_damping_{true};
    std::size_t warp_index_{0};
    std::optional<simcore::HoldMode> hold_;
};

}  // namespace simapp
