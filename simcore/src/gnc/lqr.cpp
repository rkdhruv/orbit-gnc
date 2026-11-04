#include "simcore/gnc/lqr.hpp"

#include <cmath>
#include <limits>

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <Eigen/QR>

#include "simcore/check.hpp"

namespace simcore {

namespace {

void check_inputs(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, const Eigen::MatrixXd& Q,
                  const Eigen::MatrixXd& R) {
    const Eigen::Index n = A.rows();
    const Eigen::Index m = B.cols();
    SIMCORE_CHECK(n > 0 && A.cols() == n && B.rows() == n && m > 0, "A must be n x n, B n x m");
    SIMCORE_CHECK(Q.rows() == n && Q.cols() == n && R.rows() == m && R.cols() == m,
                  "Q must be n x n and R m x m");
    SIMCORE_CHECK(A.allFinite() && B.allFinite() && Q.allFinite() && R.allFinite(),
                  "CARE inputs must be finite");
    SIMCORE_CHECK((Q - Q.transpose()).norm() <= 1e-12 * Q.norm() &&
                      (R - R.transpose()).norm() <= 1e-12 * R.norm(),
                  "Q and R must be symmetric");
    const double q_min = Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(Q).eigenvalues().minCoeff();
    SIMCORE_CHECK(q_min >= -1e-12 * Q.norm(), "Q must be positive semi-definite");
    SIMCORE_CHECK(R.llt().info() == Eigen::Success, "R must be positive definite");
}

}  // namespace

Eigen::MatrixXd solve_care(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B,
                           const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R) {
    check_inputs(A, B, Q, R);
    const Eigen::Index n = A.rows();
    const Eigen::MatrixXd G = B * R.llt().solve(B.transpose());  // B R^-1 B^T

    Eigen::MatrixXd Z(2 * n, 2 * n);
    Z << A, -G, -Q, -A.transpose();

    // Newton iteration for sign(H). Determinant scaling (c = |det Z|^(-1/2n), computed through
    // log |det| from the LU factors to avoid overflow) speeds up the early steps; it is dropped
    // once the iterates are close, where it would only slow the final quadratic convergence.
    bool converged = false;
    bool scale = true;
    double prev_change = std::numeric_limits<double>::infinity();
    for (int it = 0; it < 100; ++it) {
        const Eigen::PartialPivLU<Eigen::MatrixXd> lu(Z);
        double c = 1.0;
        if (scale) {
            const double log_det = lu.matrixLU().diagonal().cwiseAbs().array().log().sum();
            c = std::exp(-log_det / static_cast<double>(2 * n));
        }
        const Eigen::MatrixXd Z_next = 0.5 * (c * Z + lu.inverse() / c);
        const double change = (Z_next - Z).lpNorm<1>() / Z_next.lpNorm<1>();
        Z = Z_next;
        if (change < 1e-2) {
            scale = false;
        }
        // Done at 1e-14, or once the unscaled (quadratically converging) steps are small and
        // stop shrinking: that is the roundoff floor, which for a badly scaled problem can sit
        // above 1e-14.
        if (change < 1e-14 || (!scale && change < 1e-8 && change >= prev_change)) {
            converged = true;
            break;
        }
        prev_change = change;
    }
    SIMCORE_CHECK(converged, "CARE: matrix sign iteration did not converge");

    const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n, n);
    Eigen::MatrixXd M(2 * n, n);
    M << Z.topRightCorner(n, n), Z.bottomRightCorner(n, n) + I;
    Eigen::MatrixXd rhs(2 * n, n);
    rhs << -(Z.topLeftCorner(n, n) + I), -Z.bottomLeftCorner(n, n);
    const Eigen::MatrixXd P = M.colPivHouseholderQr().solve(rhs);
    return 0.5 * (P + P.transpose());  // symmetric up to roundoff; make it exactly so
}

Eigen::MatrixXd lqr_gain(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B,
                         const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R) {
    const Eigen::MatrixXd P = solve_care(A, B, Q, R);
    return R.llt().solve(B.transpose() * P);
}

}  // namespace simcore
