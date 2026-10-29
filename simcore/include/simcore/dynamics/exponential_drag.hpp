#pragma once

#include "simcore/constants.hpp"
#include "simcore/dynamics/force_model.hpp"

namespace simcore {

// Aerodynamic drag in a piecewise-exponential atmosphere (Vallado, "Fundamentals of
// Astrodynamics and Applications", Table 8-4: 28 bands from 0 to 1000 km).
//   a = -1/2 rho(h) (Cd A / m) |v_rel| v_rel,   v_rel = v - w x r
// The atmosphere co-rotates with the Earth at w about ECI z; altitude is h = |r| - R above a
// spherical Earth. Mass is fixed here; the vehicle state will supply it once mass varies (M2.3).
class ExponentialDrag final : public ForceModel {
public:
    // Throws std::invalid_argument unless cd, area and mass are finite and > 0 and the
    // atmosphere rotation rate is finite.
    ExponentialDrag(double cd, double area_m2, double mass_kg,
                    double atmosphere_rotation_radps = earth::rotation_rate_radps);

    // Density at altitude h: rho0 exp(-(h - h0) / H) in the band with base h0 <= h. Below 0 km the
    // first band and above 1000 km the last band are extrapolated. Throws for non-finite h.
    [[nodiscard]] static double density_kgpm3(double altitude_m);

    [[nodiscard]] Eigen::Vector3d acceleration_eci(const State& s) const override;

private:
    double half_b_;  // 1/2 Cd A / m [m^2/kg]
    double w_;       // atmosphere rotation rate about z [rad/s]
};

}  // namespace simcore
