# Orbit GNC

A KSP-style spacecraft flight and mission-design simulator with real guidance,
navigation and control, written from scratch in C++20. The deterministic
simulation core (`simcore`) is separate from the raylib + Dear ImGui front end
(`simapp`), and every physics feature is checked against a reference value.

> Status: **M3, interactive front end.** Orbit propagation (point mass, J2, drag), events, rigid-body
> attitude, thrusters and reaction wheels in a coupled 6-DOF sim, flown from a 3D front end with
> time warp and deterministic replay. Guidance and control (the autopilot) come in M4; see
> [ROADMAP.md](ROADMAP.md).

**Live demo:** https://rkdhruv.github.io/orbit-gnc/ (built and deployed by CI from `main`).

Controls: Shift/Ctrl throttle up/down, Z/X full/cut, W/S A/D Q/E pitch/yaw/roll (reaction wheels),
T rate damping, `,` / `.` time warp (1x to 100,000x; above 100x needs a coast on rails), mouse drag
to turn the camera, scroll to zoom. "Verify replay" re-runs the session from its input log.

## Build

Requires CMake ≥ 3.25, Ninja and a C++20 compiler. Dependencies (Eigen,
GoogleTest, raylib, Dear ImGui, rlImGui) are fetched automatically.

```sh
cmake --preset debug && cmake --build --preset debug
ctest --preset debug
./build/debug/simapp/simapp
```

| Preset    | Purpose                                              |
|-----------|------------------------------------------------------|
| `debug`   | Day-to-day development                               |
| `release` | Optimised build                                      |
| `asan`    | AddressSanitizer + UBSan, headless (no `simapp`)     |
| `web`     | Emscripten build of `simapp` to HTML/WASM            |

### Web build

Set `EMSCRIPTEN_ROOT` to the Emscripten install:

```sh
export EMSCRIPTEN_ROOT=$(brew --prefix emscripten)/libexec   # Homebrew
# export EMSCRIPTEN_ROOT=$EMSDK/upstream/emscripten          # emsdk
cmake --preset web && cmake --build --preset web
python3 -m http.server -d build/web/simapp   # then open http://localhost:8000/simapp.html
```

## Layout

```
simcore/   simulation library: no UI, no I/O, deterministic
simapp/    raylib + ImGui front end
tests/     GoogleTest suite, mirrors simcore/
cmake/     dependency pins
```

Every target is built with `-ffp-contract=off`, so the compiler never fuses
`a*b + c` into an FMA, and replays are bit-identical on the same platform.
That flag doesn't reach Eigen's hand-written arm64 SIMD kernels, which use FMA
intrinsics, so matrix products may differ in the last bit between arm64 and
x86-64.

## Results

Each row is a GoogleTest assertion; the bound in the Reference column is what the test enforces.

| Claim | Reference | Measured |
|-------|-----------|----------|
| **M1: translational core** | | |
| Circular speed at 420 km | ≈ 7.66 km/s | 7657.27 m/s |
| `Epoch`: 1 million 1 ms steps | +1000 s exactly | drift 6.7e-13 s (bare `double`: 0.047 s) |
| DP5(4) vs analytic, circular 420 km, 100 orbits (rtol 1e-12) | < 15.9 m | 0.72 m |
| Energy drift over 100 orbits, late/early max \|ΔE\| | Verlet ≤ 1.1, RK4 > 5 | Verlet 1.000, RK4 8.05 |
| Observed order: RK4 / Verlet / DP5 | 4 / 2 / 5 | 4.06 / 2.00 / 4.90 |
| DP5 vs Kepler, e = 0.7, 100 orbits (rtol 1e-12) | < 25 m | 3.5 m |
| DP5 vs Kepler, hyperbolic flyby | < 0.1 m | 8.2e-6 m |
| Periapsis event time vs Kepler, 10 orbits (rtol 1e-12) | < 1 ms | 5.65 µs |
| Impact time vs radial free fall from R + 1000 km | < 1 ms | 1.9e-10 s |
| **M2: perturbations and rigid-body spacecraft** | | |
| J2 nodal regression, 420 km, i = 51.6° | −4.951°/day first order (±1%) | −4.972°/day (1.0043×) |
| Sun-synchronous orbit, 700 km, i = 98.188° | +0.98565°/day (±1%) | 1.0044× |
| Drag decay, 400 km, B = 0.022 m²/kg, 1 day | −369.31 m (±0.1%) | −369.30 m (0.99998×) |
| Torque-free energy and inertial \|H\|, 600 s | conserved (< 1e-8 / 1e-7) | 1.9e-11 / 2.3e-10 |
| Intermediate-axis tumble growth rate, diag(100, 200, 300) | Ω/√3 (±0.5%) | 1.0016×; flips within 60 s |
| Full burn in free space, Isp 300 s, 800 → 400 kg | 2039.2355 m/s (±1e-6) | error 2.4e-10 m/s |
| Burn through burnout at a 1 s tick | no Δv without propellant (±1e-6) | error 7e-12 m/s |
| Reaction wheels on a tumbling body: total inertial H, 100 s | conserved (< 1e-9) | 1.6e-10 relative |
| **M3: interactive front end** | | |
| Sim facade vs a bare RK4 loop, driven at 60 and 144 fps | bit-identical | bit-identical (100 ticks, command change) |
| Sim clock after 100,000 ticks of 0.02 s | t0 + 2000 s | exact |
| Floating origin: 1 cm offset at 7,000 km | kept (naive float loses it) | error < 1e-9 m (naive ≥ 0.009 m) |
| HUD telemetry: GTO period, 300 × 35,786 km | 37,980.42 s | within 0.01 s |
| On rails vs RK4, 600 s coast | same motion (< 1e-3 m) | 8.9e-8 m, 4.8e-11 m/s |
| 1,000,000 ticks on rails | the Kepler solution | bit-identical |
| Rate damping stops a 0.027 rad/s tumble | within 120 s | wheels take \|Iω₀\| = 9.877 N·m·s |
| Replay of a flown session (3000 frames, burns, warp on rails) | bit-identical on the same machine | bit-identical |
| Rate damping at 10 fps and 100x warp (command held ≤ 0.5 s) | stable, reaches rails | on rails within 200 s (held a whole frame: never) |
| Impact from a 100 km drop vs the M1 event finder | stops within one tick after | t_ref ≤ t_stop < t_ref + 0.02 s |
| **M4: attitude control and autopilot** | | |
| PD 10° step (ωn = 0.1 rad/s, ζ = 0.7): overshoot / peak time | 4.5988% / 43.99 s (second order) | 4.5993% / 43.80 s |
| Prograde hold, 420 km, 300 s | body x on the velocity (< 0.01°) | 1.7e-6° |
| CARE (matrix sign function), scalar x' = x + u | P = 1 + √2 | within 1e-12 |
| LQR gains vs the per-axis closed form | k1 = √(qθ/r), k2 = √((2m√(qθr) + qω)/r) | within 1e-9 |
| Gravity-gradient pitch libration, 420 km | 2π / (n√(3(I_y − I_x)/I_z)) = 4260.44 s | 4261.74 s (0.03%: the 2° amplitude) |
| RCS couple, 10 s at full torque | ω = τt/I, no translation | v stays exactly 0; Δm = ṁt to 1e-13 kg |
| Momentum dump under gravity gradient, prograde hold, 4000 s | 3 dumps, first at 1261.1 s | 3, first at 1260.0 s; peak saturation 0.8002 |
| Dumping leaves the hold undisturbed | PD offset τ/Kp = 0.022° | 0.0228° (without dumping: wheel saturates, 140° off) |
| Dump propellant | ∫\|τ_rcs\| dt / (L·Isp·g₀) ≈ 3.24e-3 kg | 3.2402e-3 kg |
| Node: 50 m/s prograde at 420 km, impulsive prediction | apoapsis 600.505 km (vis-viva) | 600.505 km |
| Node flown by the autopilot (400 N, 99 s burn) | Δv 50 m/s; ignition t_node − 49.80 s | 50.0000 m/s; −49.50 s (0.5 s hold); apoapsis 600.410 km |
| Node 360 s late (slew not done): re-planned burn | same orbit as on time (± 2 km) | apoapsis 600.395 km (stale direction: 587.4 km) |

PD vs LQR, 30° skew-axis slew on the default vehicle (0.1 N·m wheels, I = diag(300, 400, 450) kg·m²):

| Controller | Settle (< 0.1°) | Overshoot | Wheel effort ∫Σ\|τ\| dt | Peak torque requested |
|---|---|---|---|---|
| PD, ωn = 0.05 rad/s, ζ = 0.7 | 157 s | 4.09% | 14.96 N·m·s | 0.393 N·m (saturates the 0.1 N·m wheels) |
| LQR, Bryson (30°, 0.5°/s, 0.1 N·m) | 323 s | 0.00% | 5.74 N·m·s | 0.067 N·m (never saturates) |

PD's bandwidth-chosen gains ignore the actuator: it spends the slew pinned at the wheel limit, which
makes it faster but costs 2.6× the wheel effort and overshoots. LQR's Bryson weights put the
torque limit into the design (an error of 30° asks for exactly 0.1 N·m), so it stays linear and
critically damped, at half the speed.
