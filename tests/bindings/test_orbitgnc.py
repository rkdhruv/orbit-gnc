"""M5.1: the pybind11 module `orbitgnc` against the project's reference checks.

The bindings are wrappers only, so each test is a number the C++ suite already establishes,
reached through Python: if a value, unit or argument order is lost in the wrapping, it shows up
here. Run by ctest (`bindings.python`) with the built module on PYTHONPATH; stdlib unittest only.
"""

import math
import unittest

import numpy as np

import orbitgnc as og

MU = 3.986004418e14
R_EARTH = 6.378137e6
J2 = 1.08263e-3
G0 = 9.80665
R420 = R_EARTH + 420.0e3


def circular_420(t):
    """Circular 420 km orbit, i = 51.6 deg, RAAN 30 deg, at true anomaly 0 at epoch t."""
    el = og.KeplerianElements(a_m=R420, e=0.0, i_rad=math.radians(51.6),
                              raan_rad=math.radians(30.0), argp_rad=0.0, nu_rad=0.0)
    return og.state_from_elements(el, t)


class Constants(unittest.TestCase):
    # The Earth constants in docs/CONVENTIONS.md, exactly (they are the same literals).
    def test_earth_constants(self):
        self.assertEqual(og.earth.mu, MU)
        self.assertEqual(og.earth.radius, R_EARTH)
        self.assertEqual(og.earth.j2, J2)
        self.assertEqual(og.earth.rotation_rate_radps, 7.292115e-5)
        self.assertEqual(og.g0, G0)


class EpochArithmetic(unittest.TestCase):
    def test_split_seconds_survive_the_wrapping(self):
        # whole + fractional seconds since J2000 TT; the fraction is carried into [0, 1).
        t = og.Epoch.from_tt_seconds_since_j2000(10, 2.5)
        self.assertEqual(t.whole_seconds, 12)
        self.assertEqual(t.fractional_seconds, 0.5)

    def test_add_subtract_compare(self):
        # 0.25 and 1.5 are dyadic, so these are exact (as in test_epoch).
        t0 = og.Epoch.from_tt_seconds_since_j2000(800_000_000, 0.25)
        t1 = t0 + 1.5
        self.assertEqual(t1 - t0, 1.5)
        self.assertEqual((t1 - 1.5) - t0, 0.0)
        self.assertTrue(t0 < t1)
        self.assertEqual(t1 - 1.5, t0)

    def test_whole_seconds_must_be_an_integer(self):
        # C++ deletes the floating-point overload (it would truncate); Python refuses a float.
        with self.assertRaises(TypeError):
            og.Epoch.from_tt_seconds_since_j2000(1.5)


class Orbits(unittest.TestCase):
    # 420 km circle: v = sqrt(mu / r) = 7657.27 m/s, T = 2 pi sqrt(r^3 / mu) = 5578.2227 s.
    # Elements -> state is a few roundings, so 1e-12 relative on v.
    def test_circular_420_speed_and_period(self):
        t0 = og.Epoch.from_tt_seconds_since_j2000(800_000_000, 0.25)
        s0 = circular_420(t0)
        self.assertIsInstance(s0.r_eci, np.ndarray)
        self.assertEqual(s0.r_eci.shape, (3,))
        v = np.linalg.norm(s0.v_eci)
        self.assertAlmostEqual(v / math.sqrt(MU / R420), 1.0, delta=1e-12)
        self.assertAlmostEqual(v, 7657.27, delta=0.005)
        period = 2.0 * math.pi * math.sqrt(R420**3 / MU)
        self.assertAlmostEqual(period, 5578.2227, delta=5e-5)
        # Kepler over exactly one period: M advances by 2 pi to roundoff, ~1e-15 relative, so
        # back at the start to ~r * 1e-15 * a few = ~1e-8 m; 1e-6 m is generous.
        s1 = og.kepler_propagate(s0, t0 + period)
        self.assertLess(np.linalg.norm(s1.r_eci - s0.r_eci), 1e-6)

    def test_elements_round_trip(self):
        t0 = og.Epoch.j2000()
        el = og.KeplerianElements(a_m=2.4e7, e=0.7, i_rad=0.5, raan_rad=1.0, argp_rad=2.0,
                                  nu_rad=0.3)
        back = og.elements_from_state(og.state_from_elements(el, t0))
        for name in ("a_m", "e", "i_rad", "raan_rad", "argp_rad", "nu_rad"):
            ref = getattr(el, name)
            self.assertAlmostEqual(getattr(back, name), ref, delta=1e-9 * max(1.0, abs(ref)),
                                   msg=name)

    def test_invalid_input_raises_value_error(self):
        # SIMCORE_CHECK throws std::invalid_argument, which pybind11 maps to ValueError.
        el = og.KeplerianElements(a_m=R420, e=-0.1, i_rad=0.0, raan_rad=0.0, argp_rad=0.0,
                                  nu_rad=0.0)
        with self.assertRaises(ValueError):
            og.state_from_elements(el, og.Epoch.j2000())


class Transfers(unittest.TestCase):
    # Hohmann 300 km -> 35,786 km altitude (Curtis 6.2): dv1 = 2425.732, dv2 = 1466.824,
    # total 3892.557 m/s (CONVENTIONS: ~3.89 km/s). Same formula, so 1e-9 relative.
    def test_hohmann_to_geo(self):
        r1 = R_EARTH + 300.0e3
        r2 = R_EARTH + 35_786.0e3
        el = og.KeplerianElements(a_m=r1, e=0.0, i_rad=0.0, raan_rad=0.0, argp_rad=0.0,
                                  nu_rad=0.0)
        t0 = og.Epoch.j2000()
        plan = og.hohmann_transfer(og.state_from_elements(el, t0), t0 + 100.0, r2)
        a_t = 0.5 * (r1 + r2)
        dv1 = math.sqrt(MU * (2.0 / r1 - 1.0 / a_t)) - math.sqrt(MU / r1)
        dv2 = math.sqrt(MU / r2) - math.sqrt(MU * (2.0 / r2 - 1.0 / a_t))
        self.assertAlmostEqual(plan.dv1_mps / dv1, 1.0, delta=1e-9)
        self.assertAlmostEqual(plan.dv2_mps / dv2, 1.0, delta=1e-9)
        self.assertAlmostEqual(plan.dv1_mps + plan.dv2_mps, 3892.557, delta=1e-3)
        self.assertAlmostEqual(plan.transfer_s, math.pi * math.sqrt(a_t**3 / MU),
                               delta=1e-6)
        self.assertEqual(plan.first.t, t0 + 100.0)
        self.assertEqual(plan.first.dv_rsw_mps.shape, (3,))

    # Isp 300 s, m0 / mf = 2: dv = Isp g0 ln 2 = 2039.236 m/s (CONVENTIONS: ~2,039).
    def test_rocket_equation(self):
        dv = og.rocket_delta_v_mps(300.0 * G0, 2.0, 1.0)
        self.assertAlmostEqual(dv / (300.0 * G0 * math.log(2.0)), 1.0, delta=1e-12)


class ForceModels(unittest.TestCase):
    # On the equator (z = 0) J2's acceleration is purely radial:
    #   a = -(3/2) J2 mu R^2 / r^4 x_hat = -0.0123293 m/s^2 at 420 km.
    def test_j2_on_the_equator(self):
        s = og.State(og.Epoch.j2000(), np.array([R420, 0.0, 0.0]), np.array([0.0, 7657.0, 0.0]))
        a = og.J2Gravity().acceleration_eci(s)
        ref = -1.5 * J2 * MU * R_EARTH**2 / R420**4
        self.assertAlmostEqual(a[0] / ref, 1.0, delta=1e-14)
        self.assertEqual(a[1], 0.0)
        self.assertEqual(a[2], 0.0)

    def test_force_sum_adds(self):
        s = circular_420(og.Epoch.j2000())
        pm = og.PointMassGravity()
        j2 = og.J2Gravity()
        total = og.ForceSum().add(pm).add(j2)
        np.testing.assert_allclose(total.acceleration_eci(s),
                                   pm.acceleration_eci(s) + j2.acceleration_eci(s),
                                   rtol=1e-15, atol=0.0)


class Propagation(unittest.TestCase):
    def setUp(self):
        opts = og.DpOptions()
        opts.rtol = 1e-12
        opts.atol_pos_m = 1e-6
        opts.atol_vel_mps = 1e-9
        self.dp = og.DormandPrince54(opts)
        self.t0 = og.Epoch.from_tt_seconds_since_j2000(800_000_000, 0.25)
        self.s0 = circular_420(self.t0)
        self.period = 2.0 * math.pi * math.sqrt(R420**3 / MU)

    # One orbit, 50 samples vs Kepler. Bound (test_sampled's derivation): drift is quadratic in
    # the swept angle, 15.9 m at 100 orbits -> 15.9e-4 m = 1.6 mm at one; plus <= 6e-5 m from
    # the interpolant: 2 mm.
    def test_sampled_orbit_matches_kepler(self):
        times = [self.t0 + k * self.period / 50.0 for k in range(51)]
        states = og.propagate_sampled(self.dp, og.PointMassGravity(), self.s0, times)
        self.assertEqual(len(states), 51)
        worst = max(np.linalg.norm(s.r_eci - og.kepler_propagate(self.s0, s.t).r_eci)
                    for s in states)
        self.assertLess(worst, 2e-3)
        st = self.dp.stats
        self.assertGreater(st.accepted, 0)
        # FSAL: 1 initial + 6 per attempted step + 1 for the automatic first step.
        self.assertEqual(st.force_evals, 2 + 6 * (st.accepted + st.rejected))

    def test_deterministic(self):
        # Same inputs, same machine: bit-identical (the CONVENTIONS determinism rule, through Python).
        times = [self.t0 + k * 600.0 for k in range(20)]
        f = og.ForceSum().add(og.PointMassGravity()).add(og.J2Gravity())
        a = og.propagate_sampled(self.dp, f, self.s0, times)
        b = og.propagate_sampled(self.dp, f, self.s0, times)
        for x, y in zip(a, b):
            self.assertTrue(np.array_equal(x.r_eci, y.r_eci))
            self.assertTrue(np.array_equal(x.v_eci, y.v_eci))

    def test_bad_options_raise_value_error(self):
        opts = og.DpOptions()
        opts.rtol = -1.0
        with self.assertRaises(ValueError):
            og.DormandPrince54(opts)


if __name__ == "__main__":
    unittest.main()
