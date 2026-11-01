// M3.1: fixed-timestep scheduling, decoupled from the frame rate.
//
// The front end measures real time per frame; the scheduler turns it into a whole number of sim
// ticks of fixed length h (the "accumulator" pattern):
//   acc += warp * min(frame_dt, max_frame);  n = floor(acc / h);  acc -= n h
// so the sim always advances in identical steps however irregular the frames are. Leftover time
// stays in acc (alpha = acc / h in [0, 1) is how far the display is between ticks). Two guards
// against the "spiral of death" (a slow frame demands more ticks, which makes the next frame
// slower): a real frame longer than max_frame (a debugger stall, a dragged window) is clamped,
// and at most max_ticks are run per frame, with the excess time dropped rather than owed.
//
// Exact tests use binary fractions (1/32 s ticks, 1/64 s frames), where every operation is exact.

#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>

#include "simcore/sim/tick_scheduler.hpp"

namespace {

using simcore::TickScheduler;

constexpr double kTick = 1.0 / 32.0;
constexpr double kFrame = 1.0 / 64.0;

}  // namespace

// Frames at twice the tick rate: ticks alternate 0, 1, 0, 1, ... and alpha 0.5, 0, 0.5, 0.
// 64 frames = 1 s = 32 ticks exactly.
TEST(TickScheduler, FramesFasterThanTicksAccumulate) {
    TickScheduler sched(kTick);
    long total = 0;
    for (int f = 0; f < 64; ++f) {
        const TickScheduler::FrameTicks ft = sched.ticks_for_frame(kFrame);
        EXPECT_EQ(ft.ticks, f % 2);
        EXPECT_FALSE(ft.fell_behind);
        EXPECT_EQ(sched.alpha(), f % 2 == 0 ? 0.5 : 0.0);
        total += ft.ticks;
    }
    EXPECT_EQ(total, 32);
}

// 60 fps against 50 Hz ticks for 10 s: 5 ticks per 6 frames, 500 in all. 1/60 and 0.02 aren't
// binary fractions; acc stays below 0.04 s, so each addition rounds by <= 3.5e-18 s and 600 of
// them by < 2.1e-15 s. That can only move the tick that lands exactly at t = 10 s: 499 or 500.
TEST(TickScheduler, SixtyFpsAgainstFiftyHzTicks) {
    TickScheduler sched(0.02);
    long total = 0;
    for (int f = 0; f < 600; ++f) {
        const TickScheduler::FrameTicks ft = sched.ticks_for_frame(1.0 / 60.0);
        EXPECT_GE(ft.ticks, 0);
        EXPECT_LE(ft.ticks, 1);
        EXPECT_GE(sched.alpha(), 0.0);
        EXPECT_LT(sched.alpha(), 1.0);
        total += ft.ticks;
    }
    EXPECT_GE(total, 499);
    EXPECT_LE(total, 500);
}

// Time warp scales sim time per real second: warp 10 at 1/64 s frames is 10/64 = 5/32 s of sim
// time, exactly 5 ticks every frame. Warp 0 pauses: no ticks and the leftover is kept.
TEST(TickScheduler, WarpScalesTicksAndZeroPauses) {
    TickScheduler sched(kTick);
    for (int f = 0; f < 10; ++f) {
        EXPECT_EQ(sched.ticks_for_frame(kFrame, 10.0).ticks, 5);
    }
    EXPECT_EQ(sched.ticks_for_frame(kFrame).ticks, 0);  // acc = 1/64, alpha 0.5
    EXPECT_EQ(sched.ticks_for_frame(kFrame, 0.0).ticks, 0);
    EXPECT_EQ(sched.alpha(), 0.5);
    EXPECT_EQ(sched.ticks_for_frame(kFrame).ticks, 1);  // the paused leftover still counts
}

// A 3 s stall is clamped to max_frame = 0.25 s: 0.25 / (1/32) = 8 ticks, reported as behind.
TEST(TickScheduler, LongFrameIsClamped) {
    TickScheduler sched(kTick, 0.25, 1000);
    const TickScheduler::FrameTicks ft = sched.ticks_for_frame(3.0);
    EXPECT_EQ(ft.ticks, 8);
    EXPECT_TRUE(ft.fell_behind);
    EXPECT_EQ(sched.alpha(), 0.0);
}

// Warp 1e5 at 1/64 s frames asks for 1e5 / 64 * 32 = 50,000 ticks. Capped at 1000; the rest is
// dropped (acc reset), so the next normal frame is normal again instead of owing 49,000 ticks.
TEST(TickScheduler, TickCapDropsTheExcess) {
    TickScheduler sched(kTick, 0.25, 1000);
    const TickScheduler::FrameTicks ft = sched.ticks_for_frame(kFrame, 1e5);
    EXPECT_EQ(ft.ticks, 1000);
    EXPECT_TRUE(ft.fell_behind);
    EXPECT_EQ(sched.alpha(), 0.0);
    const TickScheduler::FrameTicks next = sched.ticks_for_frame(kFrame);
    EXPECT_EQ(next.ticks, 0);
    EXPECT_FALSE(next.fell_behind);
}

TEST(TickScheduler, RejectsInvalidInput) {
    const double kNaN = std::numeric_limits<double>::quiet_NaN();
    const double kInf = std::numeric_limits<double>::infinity();
    EXPECT_THROW((void)TickScheduler(0.0), std::invalid_argument);
    EXPECT_THROW((void)TickScheduler(kNaN), std::invalid_argument);
    EXPECT_THROW((void)TickScheduler(kTick, 0.0, 1000), std::invalid_argument);   // max_frame
    EXPECT_THROW((void)TickScheduler(kTick, 0.01, 1000), std::invalid_argument);  // < one tick
    EXPECT_THROW((void)TickScheduler(kTick, 0.25, 0), std::invalid_argument);     // max_ticks
    TickScheduler sched(kTick);
    EXPECT_THROW((void)sched.ticks_for_frame(-kFrame), std::invalid_argument);
    EXPECT_THROW((void)sched.ticks_for_frame(kNaN), std::invalid_argument);
    EXPECT_THROW((void)sched.ticks_for_frame(kFrame, -1.0), std::invalid_argument);
    EXPECT_THROW((void)sched.ticks_for_frame(kFrame, kInf), std::invalid_argument);
    // A rejected frame leaves the scheduler unchanged.
    EXPECT_EQ(sched.alpha(), 0.0);
    EXPECT_EQ(sched.ticks_for_frame(0.0).ticks, 0);  // a zero-length frame is fine
}
