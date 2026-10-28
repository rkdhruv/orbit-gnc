#pragma once

namespace simcore::earth {

inline constexpr double mu = 3.986004418e14;  // gravitational parameter GM [m^3/s^2]
inline constexpr double radius = 6.378137e6;  // equatorial radius [m]
inline constexpr double j2 = 1.08263e-3;      // second zonal harmonic [-]

}  // namespace simcore::earth

namespace simcore {

inline constexpr double g0 = 9.80665;  // standard gravity, used for Isp [m/s^2]

}  // namespace simcore
