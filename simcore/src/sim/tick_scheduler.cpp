#include "simcore/sim/tick_scheduler.hpp"

#include <algorithm>
#include <cmath>

#include "simcore/check.hpp"

namespace simcore {

TickScheduler::TickScheduler(double tick_s, double max_frame_s, std::int64_t max_ticks_per_frame)
    : tick_s_(tick_s), max_frame_s_(max_frame_s), max_ticks_(max_ticks_per_frame) {
    SIMCORE_CHECK(std::isfinite(tick_s_) && tick_s_ > 0.0, "tick must be finite and > 0");
    SIMCORE_CHECK(std::isfinite(max_frame_s_) && max_frame_s_ >= tick_s_,
                  "max frame time must be finite and at least one tick");
    SIMCORE_CHECK(max_ticks_ >= 1, "max ticks per frame must be >= 1");
}

TickScheduler::FrameTicks TickScheduler::ticks_for_frame(double frame_dt_s, double warp) {
    SIMCORE_CHECK(std::isfinite(frame_dt_s) && frame_dt_s >= 0.0,
                  "frame time must be finite and >= 0");
    SIMCORE_CHECK(std::isfinite(warp) && warp >= 0.0, "warp must be finite and >= 0");
    const bool clamped = frame_dt_s > max_frame_s_;
    acc_s_ += warp * std::min(frame_dt_s, max_frame_s_);

    double n = std::floor(acc_s_ / tick_s_);
    if (n > static_cast<double>(max_ticks_)) {
        acc_s_ = 0.0;  // drop the excess instead of owing it to later frames
        return {max_ticks_, true};
    }
    acc_s_ -= n * tick_s_;
    // acc / h rounds, so the subtraction can land a hair outside [0, h): move one tick across.
    if (acc_s_ < 0.0) {
        n -= 1.0;
        acc_s_ += tick_s_;
    } else if (acc_s_ >= tick_s_) {
        n += 1.0;
        acc_s_ -= tick_s_;
        if (n > static_cast<double>(max_ticks_)) {
            acc_s_ = 0.0;
            return {max_ticks_, true};
        }
    }
    return {static_cast<std::int64_t>(n), clamped};
}

}  // namespace simcore
