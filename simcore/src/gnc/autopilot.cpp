#include "simcore/gnc/autopilot.hpp"

#include <cmath>
#include <numbers>

#include "simcore/check.hpp"
#include "simcore/gnc/transfers.hpp"

namespace simcore {

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
    const bool raising = p.dv1_mps > 0.0;
    add_node(p.first,
             BurnCutoff{raising ? BurnCutoff::Kind::kApoapsis : BurnCutoff::Kind::kPeriapsis,
                        r_target_m});
    add_circularize(raising);  // the far apsis of the transfer orbit
}

void Autopilot::clear() {
    steps_.clear();
    exec_.reset();
    engaged_ = false;
    coasting_ = false;
}

void Autopilot::engage() {
    engaged_ = !steps_.empty();
}

void Autopilot::disengage() {
    engaged_ = false;
    exec_.reset();  // the current step stays planned; it is re-started on the next engage
    coasting_ = false;
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
        burns_.push_back(exec_->achieved_dv_mps());
        exec_.reset();
        steps_.pop_front();
    }
    if (steps_.empty()) {
        engaged_ = false;  // plan complete
        return coast(cfg);
    }
    if (!exec_) {
        const State now{s.t, s.r_eci, s.v_eci};
        const Step& step = steps_.front();
        exec_.emplace(cfg, s, materialize(step, now), hold_s_, step.cutoff, mu_);
    }
    if (exec_->phase() == ManeuverExecutor::Phase::kAlign &&
        exec_->ignition_time() - s.t > coast_margin_s_) {
        coasting_ = true;  // warp-to-node: nothing to do until it is time to align
        return coast(cfg);
    }
    return exec_->command(cfg, s, gains);
}

}  // namespace simcore
