#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "simcore/constants.hpp"
#include "simcore/dynamics/force_model.hpp"
#include "simcore/dynamics/state.hpp"
#include "simcore/integrators/dormand_prince.hpp"
#include "simcore/orbits/epoch.hpp"

namespace simcore {

// Which sign changes of g count, judged in physical (forward) time whatever the integration
// direction: kRising = g goes from < 0 to >= 0 as time increases, kFalling the reverse.
enum class EventDirection : std::uint8_t { kAny, kRising, kFalling };

// An event is a sign change of the scalar g(State) along the trajectory.
struct EventSpec {
    std::string name;
    std::function<double(const State&)> g;
    EventDirection direction = EventDirection::kAny;
    bool terminal = false;  // stop integration at the first hit
};

// Periapsis: g = r . v = |r| d|r|/dt, rising (inbound -> outbound). Apoapsis is the falling one.
EventSpec periapsis_event();
// Impact: g = |r| - R, falling, terminal.
EventSpec impact_event(double body_radius_m = earth::radius);
// Crossing of altitude h above a spherical body: g = |r| - (R + h).
EventSpec altitude_crossing_event(double altitude_m,
                                  EventDirection direction = EventDirection::kAny,
                                  double body_radius_m = earth::radius);

struct EventHit {
    std::size_t event_index;  // into the events vector passed in
    State state;              // state.t is the event time
};

struct EventRun {
    State final_state;           // at t_end, or at the terminal event's root
    std::vector<EventHit> hits;  // in the order they occur along the integration
    bool terminated = false;     // a terminal event ended the run
};

namespace detail {
// Offset dt in [0, h] (or [h, 0]) of the root of g_of(dt) on one segment, given that the caller
// detected a sign change. Brent when the ends bracket a root; otherwise (fresh end values agree in
// sign: the root is on a boundary to roundoff) the end with the smaller |g|. Exposed for testing.
double locate_root_in_segment(const std::function<double(double)>& g_of, double h_s,
                              double time_tol_s);
}  // namespace detail

// Integrate with dp from s0 to t_end, detecting events on each accepted step's dense output.
// g is compared at the two ends of each step; a sign change in the requested direction is
// located with Brent to time_tol_s on g(segment.state_at(t)), at no extra force evaluations.
// Limitations: two sign changes inside one step cancel and are missed (steps must be short
// compared with the time between roots). A non-terminal root at s0 to roundoff
// (|g(s0)| <= 1e-9 s * |dg/dt|) is the initial condition and not reported; terminal roots are
// never skipped, so chained calls (one per sim tick) can't miss an impact. Throws
// std::invalid_argument for time_tol_s not finite and > 0, or an event without g.
EventRun propagate_with_events(DormandPrince54& dp, const ForceModel& f, const State& s0,
                               const Epoch& t_end, const std::vector<EventSpec>& events,
                               double time_tol_s = 1e-6);

}  // namespace simcore
