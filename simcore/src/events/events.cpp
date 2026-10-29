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

// Window (in time) within which a root at the very start counts as the initial condition.
constexpr double kStartWindow_s = 1e-9;

struct Candidate {
    double along_s;  // |time from segment start|, i.e. order along the integration
    std::size_t index;
    State state;
};

}  // namespace

namespace detail {

double locate_root_in_segment(const std::function<double(double)>& g_of, double h_s,
                              double time_tol_s) {
    const double ga = g_of(0.0);
    const double gb = g_of(h_s);
    if ((ga > 0.0) != (gb > 0.0) || ga == 0.0 || gb == 0.0) {
        return brent_root(g_of, 0.0, h_s, time_tol_s);
    }
    // The caller saw a sign change on its cached boundary values, but fresh evaluations at the
    // ends agree in sign: the root sits on a boundary to roundoff. Take the end nearer zero.
    return (std::abs(ga) <= std::abs(gb)) ? 0.0 : h_s;
}

}  // namespace detail

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
    bool first_step = true;

    const auto on_step = [&](const DenseSegment& seg) {
        const State s_end = seg.state_at(seg.t_end());
        std::vector<Candidate> found;
        for (std::size_t i = 0; i < events.size(); ++i) {
            const EventSpec& ev = events[i];
            // (not g0/g1: g0 is standard gravity in simcore, and GCC flags the shadowing)
            const double g_start = g_prev[i];
            const double g_end = ev.g(s_end);
            g_prev[i] = g_end;
            // A terminal event that starts exactly on its surface (g(s0) = 0) and moves through it
            // fires at s0: e.g. a state exactly at the body radius, falling. Give g(s0) the sign
            // opposite to g_end so crosses() sees the sign change; the direction filter still
            // applies (starting on the surface and climbing is not an impact).
            const double g_s0 = (first_step && ev.terminal && g_start == 0.0) ? -g_end : g_start;
            if (!crosses(forward ? g_s0 : g_end, forward ? g_end : g_s0, ev.direction)) {
                continue;
            }
            const double h = seg.h_s();
            // The initial condition is not an event: on the first step, a crossing whose g(s0) is
            // zero to roundoff (|g(s0)| <= 1e-9 s * |dg/dt|, slope estimated over the step) is
            // skipped. A real event just after the start (e.g. 0.5 us) has |g(s0)| far above
            // that and is kept. Terminal events are never skipped: dropping an impact would let
            // the trajectory continue through the body.
            if (first_step && !ev.terminal &&
                std::abs(g_start) <= kStartWindow_s * std::abs(g_end - g_start) / std::abs(h)) {
                continue;
            }
            const auto g_of = [&](double dt) { return ev.g(seg.state_at(seg.t_begin() + dt)); };
            const double dt = detail::locate_root_in_segment(g_of, h, time_tol_s);
            const Epoch t_root = (dt == h) ? seg.t_end() : seg.t_begin() + dt;
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
        first_step = false;
        return StepAction::kContinue;
    };

    const State end = dp.propagate(f, s0, t_end, on_step);
    run.final_state = stop_state ? *stop_state : end;
    return run;
}

}  // namespace simcore
