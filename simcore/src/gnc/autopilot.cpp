#include "simcore/gnc/autopilot.hpp"

#include <cmath>
#include <numbers>

#include "simcore/check.hpp"
#include "simcore/gnc/transfers.hpp"
#include "simcore/orbits/kepler.hpp"

namespace simcore {

namespace {

constexpr double kMinStepDv_mps = 1e-3;  // the executor's own cutoff residual

}  // namespace

Autopilot::Autopilot(double hold_s, double coast_margin_s, double mu)
    : hold_s_(hold_s), coast_margin_s_(coast_margin_s), mu_(mu) {
    SIMCORE_CHECK(std::isfinite(hold_s_) && hold_s_ > 0.0, "hold must be finite and > 0");
    SIMCORE_CHECK(std::isfinite(coast_margin_s_) && coast_margin_s_ > 0.0,
                  "coast margin must be finite and > 0");
}

void Autopilot::add_node(const ManeuverNode& node, BurnCutoff cutoff) {
    SIMCORE_CHECK(node.dv_rsw_mps.allFinite() && node.dv_rsw_mps.norm() > 0.0,
                  "node dv must be finite and nonzero");
    steps_.push_back(Step{Kind::kNode, node, cutoff});
}

void Autopilot::add_circularize(bool at_apoapsis) {
    steps_.push_back(Step{at_apoapsis ? Kind::kCircularizeApoapsis : Kind::kCircularizePeriapsis,
                          ManeuverNode{}, BurnCutoff{}});
}

void Autopilot::add_hohmann(const State& now, const Epoch& t_burn, double r_target_m) {
    const HohmannPlan p = hohmann_transfer(now, t_burn, r_target_m, mu_);
    // The far apsis of the transfer orbit is the target radius: the new apoapsis if it lies above
    // the burn point, the new periapsis if below. (Not the sign of dv1: from an ellipse a burn at
    // periapsis can lower the apoapsis toward a target that is still above it, and vice versa.)
    const double r_burn = kepler_propagate(now, t_burn, mu_).r_eci.norm();
    const bool far_is_apoapsis = r_target_m > r_burn;
    add_node(p.first, BurnCutoff{far_is_apoapsis ? BurnCutoff::Kind::kApoapsis
                                                 : BurnCutoff::Kind::kPeriapsis,
                                 r_target_m});
    add_circularize(far_is_apoapsis);
}

void Autopilot::clear() {
    steps_.clear();
    exec_.reset();
    engaged_ = false;
    coasting_ = false;
}

void Autopilot::engage() {
    engaged_ = !steps_.empty();
    aborted_ = false;
}

void Autopilot::disengage() {
    engaged_ = false;
    coasting_ = false;
    if (exec_ && exec_->phase() == ManeuverExecutor::Phase::kBurn) {
        // Stopped part-way: keep what was burnt and leave the rest as a node at the last state.
        // That node is already due, so the next engage re-plans it as a late burn centred on
        // the time it is aligned again. An apsis cutoff still ends it on the target apsis.
        Step& step = steps_.front();
        step.delivered_mps += exec_->committed_dv_mps();
        step.kind = Kind::kNode;
        step.node = exec_->remaining_node();
    }
    exec_.reset();  // otherwise the step (if any) stays planned as it was
}

ManeuverNode Autopilot::materialize(const Step& step, const State& now) const {
    switch (step.kind) {
        case Kind::kNode:
            return step.node;
        case Kind::kCircularizeApoapsis:
            return circularize_node(now, true, mu_);
        case Kind::kCircularizePeriapsis:
            return circularize_node(now, false, mu_);
    }
    return step.node;
}

std::optional<ManeuverNode> Autopilot::next_node(const State& now) const {
    if (exec_) {
        return exec_->node();
    }
    if (steps_.empty()) {
        return std::nullopt;
    }
    return materialize(steps_.front(), now);
}

std::optional<Epoch> Autopilot::coast_until() const {
    if (!coasting_ || !exec_) {
        return std::nullopt;
    }
    return exec_->ignition_time() - coast_margin_s_;
}

VehicleCommand Autopilot::coast(const VehicleConfig& cfg) {
    VehicleCommand c;
    c.throttle.assign(cfg.thrusters.size(), 0.0);
    c.wheel_torque_nm.assign(cfg.wheels.size(), 0.0);
    c.rcs.assign(cfg.rcs.size(), 0.0);
    return c;
}

VehicleCommand Autopilot::command(const VehicleConfig& cfg, const VehicleState& s,
                                  const AttitudeGains& gains) {
    coasting_ = false;
    if (!engaged_) {
        return coast(cfg);
    }
    if (exec_ && exec_->phase() == ManeuverExecutor::Phase::kDone) {
        const bool aborted = exec_->aborted();
        if (!aborted) {
            burns_.push_back(steps_.front().delivered_mps + exec_->achieved_dv_mps());
        }
        exec_.reset();
        steps_.pop_front();
        if (aborted) {
            engaged_ = false;
            aborted_ = true;
            return coast(cfg);
        }
    }
    const State now{s.t, s.r_eci, s.v_eci};
    while (!exec_ && !steps_.empty()) {
        const Step& step = steps_.front();
        const ManeuverNode node = materialize(step, now);
        if (node.dv_rsw_mps.norm() < kMinStepDv_mps) {
            if (step.delivered_mps > 0.0) {
                burns_.push_back(step.delivered_mps);  // a resumed burn that had nothing left
            }
            steps_.pop_front();  // e.g. circularising an orbit that already is circular
            continue;
        }
        exec_.emplace(cfg, s, node, hold_s_, step.cutoff, mu_);
    }
    if (!exec_) {
        engaged_ = false;  // plan complete
        return coast(cfg);
    }
    if (exec_->phase() == ManeuverExecutor::Phase::kAlign &&
        exec_->ignition_time() - s.t > coast_margin_s_) {
        coasting_ = true;  // warp-to-node: nothing to do until it is time to align
        return coast(cfg);
    }
    return exec_->command(cfg, s, gains);
}

}  // namespace simcore
