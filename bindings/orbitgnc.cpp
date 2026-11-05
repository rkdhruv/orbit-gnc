// pybind11 module `orbitgnc`: simcore's orbit, force-model, integrator and transfer API for the
// analysis notebooks. Wrappers only: every function forwards to simcore unchanged; the only code
// here adapts argument types (Python lists for std::span, keyword constructors for aggregates).
//
// Conversions: Eigen::Vector3d <-> NumPy float64 arrays of shape (3,) (pybind11/eigen.h);
// std::vector <-> list (pybind11/stl.h); std::invalid_argument (SIMCORE_CHECK) -> ValueError.

#include <pybind11/eigen.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <memory>
#include <sstream>
#include <vector>

#include "simcore/constants.hpp"
#include "simcore/dynamics/exponential_drag.hpp"
#include "simcore/dynamics/force_model.hpp"
#include "simcore/dynamics/force_sum.hpp"
#include "simcore/dynamics/j2_gravity.hpp"
#include "simcore/dynamics/point_mass_gravity.hpp"
#include "simcore/dynamics/state.hpp"
#include "simcore/gnc/maneuver.hpp"
#include "simcore/gnc/transfers.hpp"
#include "simcore/integrators/dormand_prince.hpp"
#include "simcore/orbits/elements.hpp"
#include "simcore/orbits/epoch.hpp"
#include "simcore/orbits/kepler.hpp"
#include "simcore/vehicle/thruster.hpp"

namespace py = pybind11;
using namespace py::literals;

namespace {

std::string epoch_repr(const simcore::Epoch& t) {
    std::ostringstream os;
    os.precision(17);
    os << "Epoch(whole_seconds=" << t.whole_seconds()
       << ", fractional_seconds=" << t.fractional_seconds() << ")";
    return os.str();
}

}  // namespace

PYBIND11_MODULE(orbitgnc, m) {
    using namespace simcore;
    m.doc() = "orbit-gnc simcore bindings (SI units, ECI frame, TT epochs)";

    // --- Constants ------------------------------------------------------------------------
    py::module_ e = m.def_submodule("earth", "Earth constants (WGS-84 rotation)");
    e.attr("mu") = earth::mu;
    e.attr("radius") = earth::radius;
    e.attr("j2") = earth::j2;
    e.attr("rotation_rate_radps") = earth::rotation_rate_radps;
    m.attr("g0") = g0;

    // --- Time and state -------------------------------------------------------------------
    py::class_<Epoch>(m, "Epoch", "TT seconds since J2000.0, as whole + fractional seconds")
        .def_static("j2000", &Epoch::j2000)
        // A lambda, not overload_cast: the deleted floating-point overload is a template, which
        // overload_cast can't step around. whole_s as int64 makes Python refuse a float.
        .def_static(
            "from_tt_seconds_since_j2000",
            [](std::int64_t whole_s, double frac_s) {
                return Epoch::from_tt_seconds_since_j2000(whole_s, frac_s);
            },
            "whole_s"_a, "frac_s"_a = 0.0)
        .def_property_readonly("whole_seconds", &Epoch::whole_seconds)
        .def_property_readonly("fractional_seconds", &Epoch::fractional_seconds)
        .def(py::self + double())
        .def(py::self - py::self)  // signed difference [s]; before Epoch - float so it wins
        .def(py::self - double())
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(py::self < py::self)
        .def(py::self <= py::self)
        .def(py::self > py::self)
        .def(py::self >= py::self)
        .def("__repr__", &epoch_repr);

    py::class_<State>(m, "State", "Epoch + ECI position [m] and velocity [m/s]")
        .def(py::init<Epoch, Eigen::Vector3d, Eigen::Vector3d>(), "t"_a, "r_eci"_a, "v_eci"_a)
        .def_readwrite("t", &State::t)
        .def_readwrite("r_eci", &State::r_eci)
        .def_readwrite("v_eci", &State::v_eci);

    // --- Orbits ---------------------------------------------------------------------------
    py::class_<KeplerianElements>(m, "KeplerianElements")
        .def(py::init([](double a, double ecc, double i, double raan, double argp, double nu) {
                 return KeplerianElements{a, ecc, i, raan, argp, nu};
             }),
             "a_m"_a, "e"_a, "i_rad"_a, "raan_rad"_a, "argp_rad"_a, "nu_rad"_a)
        .def_readwrite("a_m", &KeplerianElements::a_m)
        .def_readwrite("e", &KeplerianElements::e)
        .def_readwrite("i_rad", &KeplerianElements::i_rad)
        .def_readwrite("raan_rad", &KeplerianElements::raan_rad)
        .def_readwrite("argp_rad", &KeplerianElements::argp_rad)
        .def_readwrite("nu_rad", &KeplerianElements::nu_rad);

    m.def("elements_from_state", &elements_from_state, "s"_a, "mu"_a = earth::mu);
    m.def("has_conic_elements", &has_conic_elements, "s"_a, "mu"_a = earth::mu);
    m.def("state_from_elements", &state_from_elements, "el"_a, "t"_a, "mu"_a = earth::mu);
    m.def("kepler_propagate", &kepler_propagate, "s0"_a, "t"_a, "mu"_a = earth::mu);

    // --- Force models ---------------------------------------------------------------------
    // Held by shared_ptr, so one model can sit in several ForceSums (as in C++).
    py::class_<ForceModel, std::shared_ptr<ForceModel>>(m, "ForceModel")
        .def("acceleration_eci", &ForceModel::acceleration_eci, "s"_a);
    py::class_<PointMassGravity, ForceModel, std::shared_ptr<PointMassGravity>>(m,
                                                                                "PointMassGravity")
        .def(py::init<double>(), "mu"_a = earth::mu);
    py::class_<J2Gravity, ForceModel, std::shared_ptr<J2Gravity>>(m, "J2Gravity",
                                                                  "J2 perturbation only")
        .def(py::init<double, double, double>(), "mu"_a = earth::mu, "radius_m"_a = earth::radius,
             "j2"_a = earth::j2);
    py::class_<ExponentialDrag, ForceModel, std::shared_ptr<ExponentialDrag>>(m, "ExponentialDrag")
        .def(py::init<double, double, double, double>(), "cd"_a, "area_m2"_a, "mass_kg"_a,
             "atmosphere_rotation_radps"_a = earth::rotation_rate_radps);
    py::class_<ForceSum, ForceModel, std::shared_ptr<ForceSum>>(m, "ForceSum")
        .def(py::init<>())
        .def(
            "add",
            [](ForceSum& sum, std::shared_ptr<ForceModel> model) -> ForceSum& {
                return sum.add(std::move(model));
            },
            "model"_a, py::return_value_policy::reference,
            "Append a model; returns this sum (chainable)")
        .def("__len__", &ForceSum::size);

    // --- Integration ----------------------------------------------------------------------
    py::class_<DpOptions>(m, "DpOptions")
        .def(py::init<>())
        .def_readwrite("rtol", &DpOptions::rtol)
        .def_readwrite("atol_pos_m", &DpOptions::atol_pos_m)
        .def_readwrite("atol_vel_mps", &DpOptions::atol_vel_mps)
        .def_readwrite("h_initial_s", &DpOptions::h_initial_s)
        .def_readwrite("h_min_s", &DpOptions::h_min_s)
        .def_readwrite("h_max_s", &DpOptions::h_max_s)
        .def_readwrite("max_steps", &DpOptions::max_steps);
    py::class_<DpStats>(m, "DpStats")
        .def_readonly("accepted", &DpStats::accepted)
        .def_readonly("rejected", &DpStats::rejected)
        .def_readonly("force_evals", &DpStats::force_evals);
    py::class_<DormandPrince54>(m, "DormandPrince54")
        .def(py::init<DpOptions>(), "opts"_a = DpOptions{})
        .def(
            "propagate",
            [](DormandPrince54& dp, const ForceModel& f, const State& s0, const Epoch& t_end) {
                return dp.propagate(f, s0, t_end);
            },
            "f"_a, "s0"_a, "t_end"_a)
        .def_property_readonly("stats", &DormandPrince54::stats)
        .def_property_readonly("options", &DormandPrince54::options);
    m.def(
        "propagate_sampled",
        [](DormandPrince54& dp, const ForceModel& f, const State& s0,
           const std::vector<Epoch>& times) { return propagate_sampled(dp, f, s0, times); },
        "dp"_a, "f"_a, "s0"_a, "times"_a,
        "One propagate() run to times[-1], read out at each time (list of State)");

    // --- Maneuvers and transfers ----------------------------------------------------------
    py::class_<ManeuverNode>(m, "ManeuverNode")
        .def(py::init<>())
        .def_readwrite("t", &ManeuverNode::t)
        .def_readwrite("dv_rsw_mps", &ManeuverNode::dv_rsw_mps);
    py::class_<HohmannPlan>(m, "HohmannPlan")
        .def_readonly("first", &HohmannPlan::first)
        .def_readonly("second", &HohmannPlan::second)
        .def_readonly("dv1_mps", &HohmannPlan::dv1_mps)
        .def_readonly("dv2_mps", &HohmannPlan::dv2_mps)
        .def_readonly("transfer_s", &HohmannPlan::transfer_s);
    m.def("hohmann_transfer", &hohmann_transfer, "now"_a, "t_burn"_a, "r_target_m"_a,
          "mu"_a = earth::mu);
    m.def("rocket_delta_v_mps", &rocket_delta_v_mps, "exhaust_velocity_mps"_a, "m0_kg"_a,
          "mf_kg"_a);
}
