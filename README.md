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
`a*b + c` into an FMA. That keeps floating-point results reproducible across
machines.

## Results

| Claim | Reference | Measured |
|-------|-----------|----------|
| _(filled in from M1 onwards)_ | | |
