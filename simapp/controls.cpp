#include "controls.hpp"

#include <raylib.h>

#include <algorithm>

namespace simapp {

namespace {

// +1 / -1 / 0 from a pair of held keys.
double axis(KeyboardKey plus, KeyboardKey minus) {
    return (IsKeyDown(plus) ? 1.0 : 0.0) - (IsKeyDown(minus) ? 1.0 : 0.0);
}

}  // namespace

void Controls::handle_keys(double frame_dt_s) {
    constexpr double kThrottleRate = 0.5;  // per second of real time
    if (IsKeyDown(KEY_LEFT_SHIFT)) {
        throttle_ += kThrottleRate * frame_dt_s;
    }
    if (IsKeyDown(KEY_LEFT_CONTROL)) {
        throttle_ -= kThrottleRate * frame_dt_s;
    }
    if (IsKeyPressed(KEY_Z)) {
        throttle_ = 1.0;
    }
    if (IsKeyPressed(KEY_X)) {
        throttle_ = 0.0;
    }
    throttle_ = std::clamp(throttle_, 0.0, 1.0);

    // A positive torque about +y turns the nose (+x) toward -z (down); about +z, toward +y (left);
    // about +x, the left side (+y) up, which is a roll to the right.
    rotate_ = Eigen::Vector3d(axis(KEY_E, KEY_Q), axis(KEY_W, KEY_S), axis(KEY_A, KEY_D));

    if (IsKeyPressed(KEY_T)) {
        rate_damping_ = !rate_damping_;
    }
    if (IsKeyPressed(KEY_PERIOD) && warp_index_ + 1 < kWarpLevels.size()) {
        ++warp_index_;
    }
    if (IsKeyPressed(KEY_COMMA) && warp_index_ > 0) {
        --warp_index_;
    }
}

void Controls::clamp_warp(double max_warp) {
    while (warp_index_ > 0 && kWarpLevels.at(warp_index_) > max_warp) {
        --warp_index_;
    }
}

}  // namespace simapp
