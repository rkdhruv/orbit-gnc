#include "scene.hpp"

#include <raylib.h>
#include <rlgl.h>

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <vector>

#include "render_frame.hpp"
#include "simcore/constants.hpp"
#include "simcore/orbits/orbit_track.hpp"

// rlgl has no depth-only clear, so call glClear directly. raylib already links the platform's GL.
#if defined(__EMSCRIPTEN__)
#include <GLES2/gl2.h>
#elif defined(__APPLE__)
#define GL_SILENCE_DEPRECATION
#include <OpenGL/gl3.h>
#else
#include <GL/gl.h>
#endif

namespace simapp {

namespace {

constexpr double kKm = 1000.0;
constexpr double kMinDistanceM = 10.0;
constexpr double kMaxDistanceM = 1.0e8;  // 100,000 km
constexpr float kMaxPitch = 1.5F;        // just short of straight up or down
constexpr int kOrbitPoints = 256;
constexpr double kOrbitRMaxM = 1.0e9;  // open orbits are drawn out to 1 million km

Vector3 to_rl(const Eigen::Vector3f& v) {
    return Vector3{v.x(), v.y(), v.z()};
}

// Camera position for a given distance, on a sphere around the origin (render axes, y up).
Vector3 camera_position(float yaw, float pitch, float distance) {
    return Vector3{distance * std::cos(pitch) * std::sin(yaw), distance * std::sin(pitch),
                   distance * std::cos(pitch) * std::cos(yaw)};
}

Camera3D make_camera(Vector3 position) {
    Camera3D cam{};
    cam.position = position;
    cam.target = Vector3{0.0F, 0.0F, 0.0F};
    cam.up = Vector3{0.0F, 1.0F, 0.0F};
    cam.fovy = 45.0F;
    cam.projection = CAMERA_PERSPECTIVE;
    return cam;
}

// Parallels and meridians every 15 deg on a sphere of radius r centred at the current origin,
// poles along render y. (raylib's DrawSphereWires draws triangle edges, diagonals included.)
void draw_lat_long_grid(float r, Color color) {
    constexpr int kStepDeg = 15;
    constexpr int kSegments = 96;  // per full circle
    const double deg = std::numbers::pi / 180.0;
    const double dphi = 2.0 * std::numbers::pi / kSegments;
    const auto on_sphere = [r](double lat, double lon) {
        return Vector3{static_cast<float>(r * std::cos(lat) * std::cos(lon)),
                       static_cast<float>(r * std::sin(lat)),
                       static_cast<float>(-r * std::cos(lat) * std::sin(lon))};
    };
    for (int lat_deg = -90 + kStepDeg; lat_deg < 90; lat_deg += kStepDeg) {
        for (int k = 0; k < kSegments; ++k) {
            DrawLine3D(on_sphere(lat_deg * deg, k * dphi), on_sphere(lat_deg * deg, (k + 1) * dphi),
                       color);
        }
    }
    for (int lon_deg = 0; lon_deg < 360; lon_deg += kStepDeg) {
        for (int k = 0; k < kSegments / 2; ++k) {
            const double lat0 = -0.5 * std::numbers::pi + k * dphi;
            DrawLine3D(on_sphere(lat0, lon_deg * deg), on_sphere(lat0 + dphi, lon_deg * deg),
                       color);
        }
    }
}

void draw_far_pass(const simcore::Snapshot& snap, float yaw, float pitch, double distance_m) {
    const Eigen::Vector3d& origin = snap.state.r_eci;
    const Vector3 cam_pos = camera_position(yaw, pitch, static_cast<float>(distance_m / kKm));
    const Eigen::Vector3f earth = to_render(Eigen::Vector3d::Zero(), origin, kKm);
    const auto R_km = static_cast<float>(simcore::earth::radius / kKm);

    // Near plane: half the distance to the nearest thing drawn (Earth's surface or the orbit line
    // through the spacecraft). Depth resolution at distance z is ~z^2 / (near 2^24), so a near
    // plane as far out as possible keeps the grid from flickering against the sphere.
    const float to_surface =
        Eigen::Vector3f(cam_pos.x - earth.x(), cam_pos.y - earth.y(), cam_pos.z - earth.z())
            .norm() -
        R_km;
    const double near_km =
        std::max(1e-3, 0.5 * std::min(static_cast<double>(to_surface), distance_m / kKm));
    rlSetClipPlanes(near_km, 1.0e6);
    BeginMode3D(make_camera(cam_pos));

    // Earth: dark sphere plus a latitude/longitude grid turning with the Earth (about render y,
    // which is ECI z). The grid sits 0.3% above the surface to stay clear of depth fighting.
    DrawSphereEx(to_rl(earth), R_km, 48, 96, Color{20, 50, 110, 255});
    rlPushMatrix();
    rlTranslatef(earth.x(), earth.y(), earth.z());
    const double theta_deg = earth_display_angle(snap.state.t) * 180.0 / std::numbers::pi;
    rlRotatef(static_cast<float>(theta_deg), 0.0F, 1.0F, 0.0F);
    draw_lat_long_grid(1.003F * R_km, Color{90, 140, 210, 255});
    rlPopMatrix();

    // Orbit line from the osculating conic: a ring if closed, an open arc otherwise (the same
    // e < 1 test that sampled it). Drawing must not throw mid-frame: a state with no orbit plane
    // (h = 0) just has no line.
    const simcore::State orbit{snap.state.t, snap.state.r_eci, snap.state.v_eci};
    simcore::OrbitTrack track{{}, false};
    try {
        track = simcore::sample_orbit_eci(orbit, kOrbitPoints, kOrbitRMaxM);
    } catch (const std::invalid_argument&) {
        track.points.clear();
    }
    const std::vector<Eigen::Vector3d>& pts = track.points;
    const std::size_t segments = track.closed ? pts.size() : pts.size() - 1;
    for (std::size_t k = 0; k < segments && pts.size() > 1; ++k) {
        DrawLine3D(to_rl(to_render(pts[k], origin, kKm)),
                   to_rl(to_render(pts[(k + 1) % pts.size()], origin, kKm)),
                   Color{255, 200, 60, 255});
    }

    // Zoomed out, the spacecraft is sub-pixel: a marker of fixed apparent size stands in.
    if (distance_m > 500.0) {
        DrawSphere(Vector3{0.0F, 0.0F, 0.0F}, static_cast<float>(0.008 * distance_m / kKm),
                   Color{255, 255, 255, 255});
    }
    EndMode3D();
}

void draw_near_pass(const simcore::Snapshot& snap, float yaw, float pitch, double distance_m) {
    rlSetClipPlanes(0.1, 1.0e5);
    BeginMode3D(make_camera(camera_position(yaw, pitch, static_cast<float>(distance_m))));

    // Body frame -> render: rotate by M R(q) about the origin (the spacecraft's CoM).
    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
    T.block<3, 3>(0, 0) = render_rotation(snap.state.q_body_to_eci);
    rlPushMatrix();
    rlMultMatrixf(T.data());  // Eigen is column-major, as OpenGL expects

    // A 2 x 1.2 x 1.2 m bus, an engine bell along -x (the main engine thrusts along +x), and
    // two solar panels along body y. Body axes: x red, y green, z blue.
    DrawCube(Vector3{0.0F, 0.0F, 0.0F}, 2.0F, 1.2F, 1.2F, Color{200, 200, 205, 255});
    DrawCubeWires(Vector3{0.0F, 0.0F, 0.0F}, 2.0F, 1.2F, 1.2F, Color{60, 60, 70, 255});
    DrawCylinderEx(Vector3{-1.0F, 0.0F, 0.0F}, Vector3{-1.7F, 0.0F, 0.0F}, 0.2F, 0.45F, 16,
                   Color{120, 110, 100, 255});
    for (const float side : {-1.0F, 1.0F}) {
        DrawCube(Vector3{0.0F, side * 2.2F, 0.0F}, 1.0F, 3.0F, 0.05F, Color{40, 60, 140, 255});
    }
    DrawLine3D(Vector3{0.0F, 0.0F, 0.0F}, Vector3{3.0F, 0.0F, 0.0F}, RED);
    DrawLine3D(Vector3{0.0F, 0.0F, 0.0F}, Vector3{0.0F, 3.0F, 0.0F}, GREEN);
    DrawLine3D(Vector3{0.0F, 0.0F, 0.0F}, Vector3{0.0F, 0.0F, 3.0F}, BLUE);
    rlPopMatrix();
    EndMode3D();
}

}  // namespace

void Scene::handle_input(bool mouse_free) {
    if (!mouse_free) {
        return;
    }
    if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
        const Vector2 d = GetMouseDelta();
        yaw_rad_ -= 0.005F * d.x;
        pitch_rad_ = std::clamp(pitch_rad_ + 0.005F * d.y, -kMaxPitch, kMaxPitch);
    }
    // Each wheel notch zooms by 15%, so the full 10 m - 100,000 km range is ~115 notches.
    const auto wheel = static_cast<double>(GetMouseWheelMove());
    distance_m_ = std::clamp(distance_m_ * std::pow(1.15, -wheel), kMinDistanceM, kMaxDistanceM);
}

void Scene::draw(const simcore::Snapshot& snap) const {
    draw_far_pass(snap, yaw_rad_, pitch_rad_, distance_m_);
    // The far pass's depths are in kilometre units; the near pass needs its own.
    glClear(GL_DEPTH_BUFFER_BIT);
    draw_near_pass(snap, yaw_rad_, pitch_rad_, distance_m_);
    rlSetClipPlanes(RL_CULL_DISTANCE_NEAR, RL_CULL_DISTANCE_FAR);  // restore raylib's defaults
}

}  // namespace simapp
