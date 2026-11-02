#pragma once

#include <Eigen/Core>

namespace simcore {

// Stabilising solution P of the continuous algebraic Riccati equation (CARE)
//   A^T P + P A - P B R^-1 B^T P + Q = 0
// for x' = A x + B u with cost J = integral (x^T Q x + u^T R u) dt. Method: the matrix sign
// function of the Hamiltonian H = [[A, -B R^-1 B^T], [-Q, -A^T]] by the scaled Newton iteration
// Z <- (c Z + (c Z)^-1) / 2, c = |det Z|^(-1/2n); then, since the stable invariant subspace of H is
// spanned by [I; P], (sign(H) + I) [I; P] = 0 gives the least-squares system
//   [W12; W22 + I] P = -[W11 + I; W21]   (Byers 1987).
// Requires (A, B) stabilisable and (A, Q^1/2) detectable (then H has no imaginary eigenvalues).
// Throws std::invalid_argument for mismatched or non-finite shapes, Q not symmetric positive
// semi-definite, R not symmetric positive definite, or no convergence.
Eigen::MatrixXd solve_care(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B,
                           const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R);

// The LQR gain K = R^-1 B^T P (u = -K x), with P from solve_care.
Eigen::MatrixXd lqr_gain(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B,
                         const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R);

}  // namespace simcore
