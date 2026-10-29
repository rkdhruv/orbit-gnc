#pragma once

#include <array>

#include <Eigen/Core>

#include "simcore/dynamics/state.hpp"
#include "simcore/orbits/epoch.hpp"

namespace simcore {

class DormandPrince54;

// Continuous extension of one accepted Dormand-Prince step [t0, t0 + h] (h may be negative):
// Hairer's 4th-order DOPRI5 interpolant built from the step's stages, so the state anywhere in
// the step costs no extra force evaluations. With theta = (t - t0) / h,
//   y(theta) = c0 + theta (c1 + (1 - theta) (c2 + theta (c3 + (1 - theta) c4)))
// which gives y0 at theta = 0 and y0 + (y1 - y0) at theta = 1.
class DenseSegment {
public:
    using Vec6 = Eigen::Matrix<double, 6, 1>;

    [[nodiscard]] Epoch t_begin() const { return t0_; }
    [[nodiscard]] Epoch t_end() const { return t1_; }  // stored, not t0 + h, so it is exact
    [[nodiscard]] double h_s() const { return h_s_; }

    // State at t in the segment. Throws std::invalid_argument if t lies outside it by more than
    // 1e-9 |h| (a small allowance for endpoint rounding).
    [[nodiscard]] State state_at(const Epoch& t) const;

private:
    friend class DormandPrince54;
    DenseSegment(const Epoch& t0, const Epoch& t1, double h_s, const std::array<Vec6, 5>& c)
        : t0_(t0), t1_(t1), h_s_(h_s), c_(c) {}

    Epoch t0_;
    Epoch t1_;
    double h_s_;
    std::array<Vec6, 5> c_;  // interpolant coefficients over y = [r; v]
};

}  // namespace simcore
