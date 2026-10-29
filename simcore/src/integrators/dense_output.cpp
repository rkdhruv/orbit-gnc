#include "simcore/integrators/dense_output.hpp"

#include <cmath>

#include "simcore/check.hpp"

namespace simcore {

State DenseSegment::state_at(const Epoch& t) const {
    const double dt = t - t0_;
    const double slack = 1e-9 * std::abs(h_s_);
    // Position along the step in units of |h|: 0 at t0, |h| at t1, whichever way time runs.
    const double along = (h_s_ >= 0.0) ? dt : -dt;
    SIMCORE_CHECK(along >= -slack && along <= std::abs(h_s_) + slack,
                  "DenseSegment::state_at: t outside the segment");
    const double th = dt / h_s_;
    const double th1 = 1.0 - th;
    const Vec6 y = c_[0] + th * (c_[1] + th1 * (c_[2] + th * (c_[3] + th1 * c_[4])));
    return State{t, y.head<3>(), y.tail<3>()};
}

}  // namespace simcore
