#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <vector>

#include "simcore/constants.hpp"
#include "simcore/dynamics/state.hpp"
#include "simcore/gnc/attitude_control.hpp"
#include "simcore/gnc/maneuver.hpp"
#include "simcore/vehicle/vehicle.hpp"

namespace simcore {

// Flies a plan of burns on its own: a queue of steps, each flown by a ManeuverExecutor in turn.
// A step is a node (with its cutoff) or a circularisation, which is computed only when its turn
// comes, from the orbit the earlier burns actually produced, so a finite-burn error isn't carried
// forward. Between burns the autopilot coasts with a zero command until coast_margin_s before the
// next ignition (warp-to-node: in a two-body environment the sim then goes on rails), and only
// then hands control to the executor to align. Every command is meant to be submitted, so a flown
// plan is logged and replays exactly.
class Autopilot {
public:
    // hold_s: how long each command is held (the fly_with chunk). Throws std::invalid_argument for
    // hold_s or coast_margin_s not finite and > 0.
    explicit Autopilot(double hold_s, double coast_margin_s = 900.0, double mu = earth::mu);

    void add_node(const ManeuverNode& node, BurnCutoff cutoff = {});
    // Circularise at the next apoapsis or periapsis of the orbit at the time this step comes up.
    void add_circularize(bool at_apoapsis);
    // A Hohmann transfer (hohmann_transfer) from the orbit through `now` to radius r_target with
    // the first burn at t_burn: that burn cut off on the target apoapsis (raising) or periapsis
    // (lowering), then circularise at the far apsis.
    void add_hohmann(const State& now, const Epoch& t_burn, double r_target_m);
    void clear();  // also disengages

    // Start (or stop) flying the plan. Engaging an empty plan does nothing.
    void engage();
    void disengage();

    [[nodiscard]] bool engaged() const { return engaged_; }
    [[nodiscard]] bool has_plan() const { return !steps_.empty(); }
    [[nodiscard]] std::size_t steps() const { return steps_.size(); }
    // The first step as a node: a circularise step is computed from `now` (as it would be if its
    // turn came now); a step being flown reports the executor's (possibly re-planned) node.
    [[nodiscard]] std::optional<ManeuverNode> next_node(const State& now) const;
    [[nodiscard]] const ManeuverExecutor* executor() const { return exec_ ? &*exec_ : nullptr; }
    [[nodiscard]] bool coasting() const { return coasting_; }
    // While coasting: when the coast ends (ignition - coast_margin_s), so a front end warping
    // through the coast can stop there rather than overshoot it.
    [[nodiscard]] std::optional<Epoch> coast_until() const;
    // dv achieved by each completed burn, in order (from propellant used).
    [[nodiscard]] const std::vector<double>& burns_dv_mps() const { return burns_; }

    // The command for the next hold: advances the plan (starts the next step's executor, retires
    // a finished one, disengages when the plan is done) and coasts while far from ignition.
    VehicleCommand command(const VehicleConfig& cfg, const VehicleState& s,
                           const AttitudeGains& gains);

private:
    enum class Kind : std::uint8_t { kNode, kCircularizeApoapsis, kCircularizePeriapsis };
    struct Step {
        Kind kind{Kind::kNode};
        ManeuverNode node;
        BurnCutoff cutoff;
    };
    [[nodiscard]] ManeuverNode materialize(const Step& step, const State& now) const;
    static VehicleCommand coast(const VehicleConfig& cfg);

    double hold_s_;
    double coast_margin_s_;
    double mu_;
    std::deque<Step> steps_;
    std::optional<ManeuverExecutor> exec_;
    bool engaged_{false};
    bool coasting_{false};
    std::vector<double> burns_;
};

}  // namespace simcore
