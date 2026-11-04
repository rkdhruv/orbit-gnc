#pragma once

#include "simcore/constants.hpp"
#include "simcore/dynamics/state.hpp"
#include "simcore/gnc/maneuver.hpp"

namespace simcore {

// Time from s until the orbit next reaches true anomaly nu_target (in [0, T)): (M' - M) / n with
// M the mean anomaly and n = sqrt(mu / a^3), wrapped forward. For a circular orbit nu is the
// argument of latitude (as elements_from_state reports it). Throws std::invalid_argument for an
// orbit that isn't elliptic (or has no conic elements).
double time_to_true_anomaly_s(const State& s, double nu_target_rad, double mu = earth::mu);

// A coplanar Hohmann transfer from the orbit through `now` to a circle of radius r_target, with
// the first burn at t_burn. The velocity there must be horizontal (a circular orbit, or an
// apsis): |r_hat . v_hat| <= 1e-3. With r1 = |r(t_burn)|, v1 = |v(t_burn)| and the transfer
// ellipse a_t = (r1 + r2) / 2 (Curtis 6.2):
//   dv1 = sqrt(mu (2 / r1 - 1 / a_t)) - v1,  dv2 = sqrt(mu / r2) - sqrt(mu (2 / r2 - 1 / a_t)),
//   transfer time pi sqrt(a_t^3 / mu); both burns along-track (negative = retrograde, lowering).
struct HohmannPlan {
    ManeuverNode first;
    ManeuverNode second;
    double dv1_mps{0.0};
    double dv2_mps{0.0};
    double transfer_s{0.0};
};
HohmannPlan hohmann_transfer(const State& now, const Epoch& t_burn, double r_target_m,
                             double mu = earth::mu);

// Circularise at the next apoapsis (at_apoapsis) or periapsis: dv = sqrt(mu / r) - v there,
// along-track. Throws for an orbit that isn't elliptic.
ManeuverNode circularize_node(const State& now, bool at_apoapsis, double mu = earth::mu);

// Change the inclination to i_target at the next equator crossing (ascending or descending node,
// whichever comes first): v is rotated about r_hat by the inclination change. That keeps its size
// and its radial part and turns only the along-track speed v_S (< |v| on an eccentric orbit away
// from the apsides), so dv_rsw = (0, v_S (cos di - 1), +-v_S sin di), |dv| = 2 v_S sin(di / 2);
// the sign is the one that reaches i_target. (It takes the first node, not necessarily the
// cheaper one: the node farther from Earth has the smaller v_S.) Throws for an orbit with no line
// of nodes (equatorial, i < 1e-6 rad) or one that isn't elliptic.
ManeuverNode plane_change_node(const State& now, double i_target_rad, double mu = earth::mu);

}  // namespace simcore
