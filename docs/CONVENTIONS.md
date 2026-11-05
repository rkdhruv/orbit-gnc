# Conventions

The rules the code and tests hold to. Comments in the source that say "see CONVENTIONS" mean
this file.

## Architecture

- `simcore/` is the simulation library: no raylib, ImGui, Python or I/O dependencies, and
  deterministic (no wall-clock time, no unseeded randomness; a Monte Carlo RNG is passed in
  explicitly).
- `simapp/` reads simulation state and sends commands; it never mutates sim internals directly.
- `bindings/` holds pybind11 wrappers only, no logic.
- `tests/` mirrors `simcore/`. Physics enters the integrators through the `ForceModel` interface.

## Units and frames

- SI internally: metres, seconds, kilograms, radians. Conversion happens only at the UI.
- `double` everywhere in `simcore`; world positions are never stored as `float`.
- Frames are explicit in names: `r_eci`, `v_eci`, `w_body`, `dv_rsw`.
- Quaternions use the Hamilton convention, written scalar-first, body-to-inertial
  (`q_body_to_eci`). They are renormalised after each integration step. Angular velocity is in
  the body frame.
- Absolute time is an `Epoch` (TT, whole + fractional seconds since J2000.0), never a bare
  `double`.

## Earth constants

| Quantity | Value |
|---|---|
| Gravitational parameter μ | 3.986004418e14 m³/s² |
| Equatorial radius R | 6.378137e6 m |
| J2 | 1.08263e-3 |
| Standard gravity g₀ | 9.80665 m/s² |
| Rotation rate ω⊕ (WGS-84) | 7.292115e-5 rad/s |

## Determinism

- Every target is built with `-ffp-contract=off`, so the compiler never fuses `a*b + c` into an
  FMA, and replays with the same inputs are bit-identical on the same platform.
- Eigen's hand-written arm64 SIMD kernels use FMA intrinsics regardless, so arm64 and x86-64
  results can differ in the last bit. Tests never assert bit-identity across architectures.

## Testing

- Any physics or GNC change starts with a failing test. Each assertion carries its derivation
  and a justified tolerance.
- Every feature gets a claim and a measured check; these feed the README Results table.
- A test is never weakened or deleted to make it pass; a failure is reported instead.

Reference checks the suite reproduces:

| Check | Reference |
|---|---|
| Circular 420 km orbit | period 2π√(r³/μ) = 5578.2 s (92.97 min), speed √(μ/r) ≈ 7.66 km/s |
| Hohmann 300 km → 35,786 km altitude | total Δv ≈ 3.89 km/s |
| Isp 300 s, m₀/m_f = 2 | Δv = Isp·g₀·ln 2 ≈ 2,039 m/s |
| Torque-free rigid body | energy and \|H\| conserved; intermediate-axis tumble |
| Periapsis event time vs. Kepler | < 1 ms |
| EKF (6-state, 100 runs) | average NEES within ≈ 5.3–6.7 |
| Replay with the same inputs, same machine | bit-identical |
