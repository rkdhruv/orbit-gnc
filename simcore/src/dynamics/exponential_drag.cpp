#include "simcore/dynamics/exponential_drag.hpp"

#include <array>
#include <cmath>

#include <Eigen/Geometry>  // cross(): declared in Core, defined in Geometry

#include "simcore/check.hpp"

namespace simcore {

namespace {

struct Band {
    double h0_m;  // base altitude
    double rho0;  // density at the base [kg/m^3]
    double H_m;   // scale height
};

// Vallado Table 8-4 (base altitude km, base density kg/m^3, scale height km), in metres.
constexpr std::array<Band, 28> kBands{{
    {0e3, 1.225, 7.249e3},         {25e3, 3.899e-2, 6.349e3},    {30e3, 1.774e-2, 6.682e3},
    {40e3, 3.972e-3, 7.554e3},     {50e3, 1.057e-3, 8.382e3},    {60e3, 3.206e-4, 7.714e3},
    {70e3, 8.770e-5, 6.549e3},     {80e3, 1.905e-5, 5.799e3},    {90e3, 3.396e-6, 5.382e3},
    {100e3, 5.297e-7, 5.877e3},    {110e3, 9.661e-8, 7.263e3},   {120e3, 2.438e-8, 9.473e3},
    {130e3, 8.484e-9, 12.636e3},   {140e3, 3.845e-9, 16.149e3},  {150e3, 2.070e-9, 22.523e3},
    {180e3, 5.464e-10, 29.740e3},  {200e3, 2.789e-10, 37.105e3}, {250e3, 7.248e-11, 45.546e3},
    {300e3, 2.418e-11, 53.628e3},  {350e3, 9.518e-12, 53.298e3}, {400e3, 3.725e-12, 58.515e3},
    {450e3, 1.585e-12, 60.828e3},  {500e3, 6.967e-13, 63.822e3}, {600e3, 1.454e-13, 71.835e3},
    {700e3, 3.614e-14, 88.667e3},  {800e3, 1.170e-14, 124.64e3}, {900e3, 5.245e-15, 181.05e3},
    {1000e3, 3.019e-15, 268.00e3},
}};

}  // namespace

ExponentialDrag::ExponentialDrag(double cd, double area_m2, double mass_kg,
                                 double atmosphere_rotation_radps)
    : w_(atmosphere_rotation_radps) {
    SIMCORE_CHECK(std::isfinite(cd) && cd > 0.0, "cd must be finite and > 0");
    SIMCORE_CHECK(std::isfinite(area_m2) && area_m2 > 0.0, "area must be finite and > 0");
    SIMCORE_CHECK(std::isfinite(mass_kg) && mass_kg > 0.0, "mass must be finite and > 0");
    SIMCORE_CHECK(std::isfinite(w_), "atmosphere rotation rate must be finite");
    half_b_ = 0.5 * cd * area_m2 / mass_kg;
}

double ExponentialDrag::density_kgpm3(double altitude_m) {
    SIMCORE_CHECK(std::isfinite(altitude_m), "altitude must be finite");
    // Highest band whose base is at or below h (the first band below 0 km). 28 bands: a linear
    // scan from the top is simple and deterministic.
    std::size_t k = kBands.size() - 1;
    while (k > 0 && altitude_m < kBands.at(k).h0_m) {
        --k;
    }
    const Band& b = kBands.at(k);
    return b.rho0 * std::exp(-(altitude_m - b.h0_m) / b.H_m);
}

Eigen::Vector3d ExponentialDrag::acceleration_eci(const State& s) const {
    // Velocity relative to the co-rotating air: v_rel = v - w x r, w = (0, 0, w).
    const Eigen::Vector3d v_rel = s.v_eci - Eigen::Vector3d(0.0, 0.0, w_).cross(s.r_eci);
    const double rho = density_kgpm3(s.r_eci.norm() - earth::radius);
    return (-half_b_ * rho * v_rel.norm()) * v_rel;
}

}  // namespace simcore
