#include "simcore/events/events.hpp"

#include <algorithm>
#include <cmath>
#include <optional>

#include "simcore/check.hpp"
#include "simcore/events/brent.hpp"

namespace simcore {

EventSpec periapsis_event() {
    return {"periapsis", [](const State& s) { return s.r_eci.dot(s.v_eci); },
            EventDirection::kRising, false};
}

EventSpec impact_event(double body_radius_m) {
    return {"impact", [body_radius_m](const State& s) { return s.r_eci.norm() - body_radius_m; },
            EventDirection::kFalling, true};
}

EventSpec altitude_crossing_event(double altitude_m, EventDirection direction,
                                  double body_radius_m) {
    const double r_m = body_radius_m + altitude_m;
    return {"altitude crossing", [r_m](const State& s) { return s.r_eci.norm() - r_m; }, direction,
            false};
}

namespace {

// Does a change from g_early to g_late (in physical time order) count as a crossing?
bool crosses(double g_early, double g_late, EventDirection dir) {
    const bool rising = g_early < 0.0 && g_late >= 0.0;
    const bool falling = g_early > 0.0 && g_late <= 0.0;
    switch (dir) {
        case EventDirection::kRising:
            return rising;
        case EventDirection::kFalling:
            return falling;
        case EventDirection::kAny:
            return rising || falling;
    }
    return false;
}

struct Candidate {
    double along_s;  // |time from segment start|, i.e. order along the integration
    std::size_t index;
    State state;
};

}  // namespace

EventRun propagate_with_events(DormandPrince54& dp, const ForceModel& f, const State& s0,
                               const Epoch& t_end, const std::vector<EventSpec>& events,
                               double time_tol_s) {
    SIMCORE_CHECK(std::isfinite(time_tol_s) && time_tol_s > 0.0,
                  "time_tol_s must be finite and > 0");
    for (const EventSpec& e : events) {
        SIMCORE_CHECK(static_cast<bool>(e.g), "event has no g function");
    }

    EventRun run;
    const bool forward = (t_end - s0.t) >= 0.0;
    // g at the start of the current segment. Cached from the previous segment's end so each
    // boundary is evaluated once (a root on a boundary can't be counted twice).
    std::vector<double> g_prev(events.size());
    for (std::size_t i = 0; i < events.size(); ++i) {
        g_prev[i] = events[i].g(s0);
    }
    std::optional<State> stop_state;

    const auto on_step = [&](const DenseSegment& seg) {
        const State s_end = seg.state_at(seg.t_end());
        std::vector<Candidate> found;
        for (std::size_t i = 0; i < events.size(); ++i) {
            const EventSpec& ev = events[i];
            const double g0 = g_prev[i];
            const double g1 = ev.g(s_end);
            g_prev[i] = g1;
            if (!crosses(forward ? g0 : g1, forward ? g1 : g0, ev.direction)) {
                continue;
            }
            // Root of g along the segment, as a function of the offset dt from its start.
            const double h = seg.h_s();
            const auto g_of = [&](double dt) { return ev.g(seg.state_at(seg.t_begin() + dt)); };
            const double ga = g_of(0.0);
            const double gb = g_of(h);
            double dt = 0.0;
            if ((ga > 0.0) != (gb > 0.0) || ga == 0.0 || gb == 0.0) {
                dt = brent_root(g_of, 0.0, h, time_tol_s);
            } else {
                // Sign change detected on cached values but not on fresh ones: the root sits on
                // a boundary to roundoff. Take the endpoint with the smaller |g|.
                dt = (std::abs(ga) <= std::abs(gb)) ? 0.0 : h;
            }
            const Epoch t_root = (dt == h) ? seg.t_end() : seg.t_begin() + dt;
            if (std::abs(t_root - s0.t) <= time_tol_s) {
                continue;  // the initial condition, not an event
            }
            found.push_back({std::abs(dt), i, seg.state_at(t_root)});
        }
        // Several roots in one step: report them in the order they occur, ties by event index.
        std::ranges::stable_sort(found, {}, &Candidate::along_s);
        for (const Candidate& c : found) {
            run.hits.push_back({c.index, c.state});
            if (events[c.index].terminal) {
                run.terminated = true;
                stop_state = c.state;
                return StepAction::kStop;
            }
        }
        return StepAction::kContinue;
    };

    const State end = dp.propagate(f, s0, t_end, on_step);
    run.final_state = stop_state ? *stop_state : end;
    return run;
}

}  // namespace simcore
