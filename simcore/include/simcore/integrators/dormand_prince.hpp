#pragma once

#include <cstdint>
#include <functional>
#include <limits>

#include "simcore/dynamics/force_model.hpp"
#include "simcore/dynamics/state.hpp"
#include "simcore/integrators/dense_output.hpp"
#include "simcore/orbits/epoch.hpp"

namespace simcore {

struct DpOptions {
    double rtol = 1e-10;         // relative tolerance [-]
    double atol_pos_m = 1e-3;    // absolute tolerance on each position component [m]
    double atol_vel_mps = 1e-6;  // absolute tolerance on each velocity component [m/s]
    double h_initial_s = 0.0;    // 0 = automatic (Hairer's starting-step heuristic)
    double h_min_s = 1e-6;       // a controller step below this throws (step-size collapse)
    double h_max_s = std::numeric_limits<double>::infinity();
    long max_steps = 10'000'000;  // attempted steps (accepted + rejected); exceeding it throws
};

struct DpStats {
    long accepted = 0;
    long rejected = 0;
    // FSAL: 1 initial + 6 per attempted step, plus 1 when h_initial_s is automatic.
    long force_evals = 0;
};

enum class StepAction : std::uint8_t { kContinue, kStop };

// Adaptive Dormand-Prince 5(4) integrator with error control and dense output.
//
// Each step evaluates 7 stages; the 5th-order solution is propagated and its difference from the
// embedded 4th-order one estimates the local error. The 7th stage is f(t + h, y1), reused as the
// next step's first stage (FSAL), so a step costs 6 force evaluations.
class DormandPrince54 {
public:
    // Validates the options; throws std::invalid_argument on bad values.
    explicit DormandPrince54(DpOptions opts = {});

    // Integrate s0 -> t_end (forward or backward). Never steps past t_end: the final step is
    // clipped and the returned state's t is exactly t_end. on_step (optional) is called after
    // every accepted step with its dense segment; returning kStop ends integration early and the
    // state at that segment's end is returned. Resets stats() at the start of each call.
    State propagate(const ForceModel& f, const State& s0, const Epoch& t_end,
                    const std::function<StepAction(const DenseSegment&)>& on_step = {});

    [[nodiscard]] const DpStats& stats() const { return stats_; }
    [[nodiscard]] const DpOptions& options() const { return opts_; }

private:
    DpOptions opts_;
    DpStats stats_;
};

}  // namespace simcore
