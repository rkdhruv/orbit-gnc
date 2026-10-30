# Orbit GNC

A KSP-style spacecraft flight and mission-design simulator with real guidance,
navigation and control, written from scratch in C++20. The deterministic
simulation core (`simcore`) is separate from the raylib + Dear ImGui front end
(`simapp`), and every physics feature is checked against a reference value.

> Status: **M0, project skeleton.** There is no physics yet.

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
