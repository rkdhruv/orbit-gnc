#pragma once

#include <functional>

namespace simcore {

// Root of g in the bracket [a, b] (either order) by Brent's method (Brent 1973, "zeroin").
//
// Each iteration tries inverse quadratic interpolation through the last three points (or the
// secant through two), which converges superlinearly near a simple root, and falls back to
// bisection whenever that step would leave the bracket or fails to shrink it fast enough. The
// bracket [b, c] always holds a sign change, so the result can't wander off.
//
// Requires finite a, b, g(a), g(b) with g(a) and g(b) not of the same sign, and finite xtol > 0.
// Returns x with |x - x*| <= max(xtol, 4 eps |x|) for a sign change x* in the bracket; a root
// exactly at an endpoint is returned exactly. Throws std::invalid_argument on bad input, a
// non-finite g, or more than max_iter iterations.
double brent_root(const std::function<double(double)>& g, double a, double b, double xtol,
                  int max_iter = 100);

}  // namespace simcore
