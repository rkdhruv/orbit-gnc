#pragma once

#include <cstdint>

namespace simcore {

// Turns real frame times into a whole number of fixed sim ticks (the accumulator pattern), so the
// sim always advances in identical steps however irregular the frames are:
//   acc += warp * min(frame_dt, max_frame);   n = floor(acc / h);   acc -= n h
// The leftover acc in [0, h) carries to the next frame; alpha() = acc / h is how far real time is
// between ticks (for interpolating the display). Two guards against the "spiral of death" (a slow
// frame asks for more ticks, which makes the next frame slower): real frames longer than
// max_frame are clamped, and at most max_ticks run per frame, with the excess time dropped rather
// than owed. Either case sets fell_behind. Pure arithmetic: the caller measures the frame time.
class TickScheduler {
public:
    struct FrameTicks {
        std::int64_t ticks;
        bool fell_behind;  // sim time was dropped this frame (clamped frame or tick cap)
    };

    // Throws std::invalid_argument unless tick_s is finite and > 0, max_frame_s is finite and
    // >= tick_s, and max_ticks_per_frame >= 1.
    explicit TickScheduler(double tick_s, double max_frame_s = 0.25,
                           std::int64_t max_ticks_per_frame = 1000);

    // frame_dt_s: real seconds since the last frame, finite and >= 0. warp: sim seconds per real
    // second, finite and >= 0 (0 pauses). Throws std::invalid_argument, leaving the state
    // unchanged.
    FrameTicks ticks_for_frame(double frame_dt_s, double warp = 1.0);

    [[nodiscard]] double alpha() const { return acc_s_ / tick_s_; }  // in [0, 1)
    [[nodiscard]] double tick_s() const { return tick_s_; }

private:
    double tick_s_;
    double max_frame_s_;
    std::int64_t max_ticks_;
    double acc_s_{0.0};  // sim time owed, in [0, tick_s)
};

}  // namespace simcore
