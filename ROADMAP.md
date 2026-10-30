# Roadmap

Each milestone ends with measurable "done when" checks. Every check becomes a row in the README
Results table, with the reference value and the measured number. Tasks inside a milestone run
through the test-first loop in CLAUDE.md.

## M0: Project skeleton (done)
- [x] CMake targets `simcore`, `simapp`, `simcore_tests`; presets debug, release, asan, web
- [x] Eigen, GoogleTest, raylib, Dear ImGui, rlImGui via FetchContent (SHA256-pinned)
- [x] clang-format, clang-tidy, `-ffp-contract=off`; CI on macOS arm64 + Linux + ASan/UBSan
- **Done when:** CI green on both platforms with one trivial test.

## M1: Translational core
- [x] `State`, `Epoch` (int64 s + fraction, TT), `ForceModel`; point-mass gravity
- [x] Velocity Verlet, RK4, adaptive Dormand–Prince 5(4) with dense output
- [x] Event detection with Brent root-finding: periapsis, impact, altitude crossing
- [x] State ↔ classical elements (circular/equatorial conventions); Kepler propagation, elliptic and hyperbolic
- **Done when:**
  - analytic and numerical propagation agree over 100 orbits (e = 0.7: < 25 m at rtol 1e-12);
  - detected periapsis times match Kepler to < 1 ms. This holds at rtol 1e-12; at default tolerances the error passes 1 ms by orbit 10.

## M2: Perturbations and rigid-body spacecraft
- [x] J2 and an exponential-atmosphere drag model as `ForceModel`s
  - J2 is needed for realistic LEO from M3 onward; drag for decay and re-entry.
  - Check J2 against the analytic secular RAAN drift, dΩ/dt = −1.5 n J2 (R/p)² cos i.
- [x] Quaternion attitude (Hamilton, body-to-inertial), Euler's equations with the gyroscopic term ω × (Iω), inertia tensor
- [x] Thrusters with mass flow ṁ = −T/(Isp g₀)
  - Reaction wheels with torque and speed limits, with wheel momentum in Euler's equations: I ω̇ = τ − ω × (Iω + h_w) − ḣ_w.
  - Mass and inertia update as fuel burns; tank at the CoM is a stated assumption.
- [x] Coupled 6-DOF integration
  - A `VehicleState` alongside `State`, so the M1 translational API stays unchanged.
  - Fixed-step RK4 on [r, v, q, w, m_p, h], matching M3's fixed sim tick; q renormalised and m_p
    clamped after each step; thrusters on or off per step. Dormand-Prince stays for coasting.
  - Simplifications: no jet damping or I_dot w term; drag keeps its own fixed mass.
- **Done when:**
  - torque-free spin conserves energy and |H|, and the intermediate-axis tumble appears;
  - a full burn in free space matches the rocket equation (Isp 300 s, m0/mf = 2 → 2,039 m/s);
  - J2 RAAN drift matches the analytic rate.

## M3: Interactive front end
- [ ] Sim facade: fixed-timestep sim loop decoupled from the frame rate
  - `simapp` sends commands through a queue and reads snapshots.
  - It never touches internals; this is the architecture rule made concrete.
- [ ] raylib 3D scene: Earth, spacecraft model with attitude, orbit line
  - Floating origin: positions stay double in simcore, and only camera-relative offsets become float.
- [ ] Dear ImGui HUD: orbit elements, attitude, fuel, Δv remaining (rocket equation), wheel momentum
- [ ] Manual control: throttle, RCS rotation; time warp
  - On rails (Kepler) while coasting with perturbations off; integrated while thrusting or when J2/drag are on.
  - Switching between the two must be continuous.
- [ ] Deterministic replay: record timestamped inputs per sim tick, re-run identically
- [ ] Emscripten build deployed to GitHub Pages from CI
  - Enable exception catching (`-sDISABLE_EXCEPTION_CATCHING=0`) so `SIMCORE_CHECK` failures can be shown in the browser rather than aborting it.
- **Done when:**
  - you can fly a manual orbit raise in the browser;
  - a replay is bit-identical on the same machine.
  - Native and wasm need not match bit-for-bit; see the Eigen FMA note in CLAUDE.md.

## M4: Attitude control and autopilot
- [ ] PD attitude controller on the quaternion error, then LQR linearised about the hold attitude
  - Hold modes: prograde, retrograde, normal, radial.
- [ ] Wheel momentum management: desaturation with thrusters when wheels approach their limit
- [ ] Maneuver nodes with the predicted post-burn orbit
  - The autopilot executes them: point, wait, burn centred on the node, cut off on achieved Δv.
- [ ] Hohmann and plane-change calculators that create nodes
- **Done when:**
  - the autopilot flies LEO (300 km) → GEO on its own;
  - total Δv is within 2% of the impulsive 3.89 km/s, with the finite-burn loss reported;
  - final orbit: altitude within 10 km of GEO, e < 1e-3;
  - a PD vs LQR comparison of slew time, overshoot and wheel effort.

## M5: Analysis tooling and mission design
- [ ] pybind11 module (`bindings/`, wrappers only) + Jupyter notebook skeleton in `analysis/`
- [ ] Moon as a third-body `ForceModel` on a deterministic analytic orbit
  - Spheres of influence (Earth–Moon ≈ 66,000 km); patched-conic transitions via event detection.
- [ ] Lambert solver (Izzo 2015); porkchop plot in the notebook
- [ ] Lunar transfer planned with patched conics, then flown in full N-body
- **Done when:**
  - Lambert reproduces a published example (Curtis Example 5.2) and Izzo's multi-revolution cases;
  - the drift of the patched-conic plan in N-body is quantified.

## M6: Rendezvous
- [ ] A second, target spacecraft; relative state in the target's LVLH frame
- [ ] Clohessy–Wiltshire two-burn rendezvous planner (circular target orbit); closing-approach guidance
- [ ] CW predictions vs full nonlinear propagation at growing separations
- **Done when:**
  - the autopilot brings the chaser to within 10 m and 0.1 m/s of the target from 10 km behind;
  - the CW error vs separation curve is plotted.

## M7: Navigation
- [ ] Time scales and frames
  - UTC ↔ TT with a fixed leap-second table, Julian Date.
  - ECEF ↔ ECI via Earth rotation (GMST); ground-station positions.
- [ ] Monte Carlo harness in `mc/` with the seeded RNG passed in explicitly
  - Needed here for the 100-run consistency test, and reused in M8.
- [ ] Ground stations with visibility; range and range-rate measurements with noise
- [ ] EKF with the state transition matrix
  - Gravity-gradient Jacobian G = −μ/r³ (I − 3 r̂r̂ᵀ), plus J2 partials if J2 is on.
  - Process noise tuned; NEES/NIS tests.
- [ ] Navigation in the loop: guidance uses the estimated state, not the truth
  - For rendezvous, add a relative range/bearing sensor to the target.
- **Done when:**
  - average NEES over 100 runs stays within its 95% chi-square bounds (6-state: ≈ 5.3–6.7);
  - rendezvous still succeeds using the estimated state.

## M8: Monte Carlo dispersions, external validation and write-up
- [ ] Dispersions on the M7 harness: burn errors, sensor noise, initial uncertainty → miss-distance and Δv statistics
- [ ] External validation: propagate a published state and compare against JPL Horizons over a stated arc
  - Use J2 + Sun/Moon. Choose a MEO/GNSS satellite or a lunar probe, where drag doesn't dominate.
  - Cross-check one scenario in GMAT.
- [ ] Technical report (10–15 pages, LaTeX): models, controllers, filter, validation, limitations
- [ ] README: Results table, GIFs, live demo link, architecture diagram
- **Done when:** every claim in the README Results table has a measured number.

## Interview prep (one or two per milestone)
- **M1:**
  - Why int64 + fraction for time?
  - Why is Verlet's energy error bounded while RK4's drifts?
  - How does dense output let events be found without extra force evaluations?
- **M2:** Why does the intermediate axis tumble? Why quaternions over Euler angles?
- **M3:** Why does a fixed sim timestep matter for replay? What is a floating origin for?
- **M4:** PD vs LQR: what does the LQR weighting trade off? Where do finite-burn losses come from?
- **M5:** What does Lambert's problem solve? Why do patched conics drift in N-body?
- **M6:** When do the CW equations break down, and why?
- **M7:** What do NEES and NIS test? What does it mean when NEES is too high?
- **M8:** What dominates the Horizons residual, and how would you reduce it?
