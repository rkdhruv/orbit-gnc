#include "hud.hpp"

#include <imgui.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <numbers>

#include "simcore/version.hpp"

namespace simapp {

namespace {

constexpr double kRadToDeg = 180.0 / std::numbers::pi;

// An angle in [0, 2 pi) as degrees for "%.4f": values within rounding of 360 print as 0, not 360.
double wrap_deg(double rad) {
    const double deg = rad * kRadToDeg;
    return deg >= 359.99995 ? 0.0 : deg;
}

// "h:mm:ss" for a non-negative duration.
void text_duration(const char* label, double seconds) {
    const auto total = static_cast<long long>(seconds);
    ImGui::Text("%-14s %lld:%02lld:%02lld", label, total / 3600, (total / 60) % 60, total % 60);
}

// Altitude in km, "none" for the apoapsis of an open orbit, "n/a" without conic elements.
void text_km(const char* label, double metres) {
    if (std::isnan(metres)) {
        ImGui::Text("%-14s n/a", label);
    } else if (std::isinf(metres)) {
        ImGui::Text("%-14s none (escape)", label);
    } else {
        ImGui::Text("%-14s %.3f km", label, metres / 1000.0);
    }
}

}  // namespace

HudAction draw_hud(const simcore::Telemetry& tm, const simcore::VehicleConfig& cfg,
                   const HudFrame& f) {
    HudAction action = HudAction::kNone;
    ImGui::SetNextWindowPos(ImVec2(16.0F, 16.0F), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(390.0F, 0.0F), ImGuiCond_FirstUseEver);
    ImGui::Begin("Orbit GNC");
    text_duration("Mission time", f.mission_time_s);

    if (ImGui::CollapsingHeader("Flight", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Text("%-14s %.0fx  %s", "Time warp", f.warp,
                    f.on_rails ? "(on rails: exact Kepler)" : "(integrating)");
        std::array<char, 32> thr{};
        std::snprintf(thr.data(), thr.size(), "throttle %.0f%%", 100.0 * f.throttle);
        ImGui::ProgressBar(static_cast<float>(f.throttle), ImVec2(-1.0F, 0.0F), thr.data());
        ImGui::Text("%-14s %s", "Rate damping",
                    f.hold ? "off (hold active)" : (f.rate_damping ? "on" : "off"));
        static constexpr std::array<const char*, 6> kHoldNames{
            "prograde", "retrograde", "normal", "antinormal", "radial out", "radial in"};
        ImGui::Text("%-14s %s", "Attitude hold",
                    f.hold ? kHoldNames.at(static_cast<std::size_t>(*f.hold)) : "off");
        ImGui::Text("%-14s %s", "Controller",
                    f.use_lqr ? "LQR (Bryson: 30 deg, 0.5 deg/s, wheel limit)"
                              : "PD (wn 0.05 rad/s, zeta 0.7)");
        ImGui::SameLine();
        if (ImGui::SmallButton(f.use_lqr ? "use PD" : "use LQR")) {
            action = HudAction::kToggleController;
        }
        static constexpr std::array<const char*, 7> kHoldButtons{"Off",  "Pro",   "Retro", "Norm",
                                                                 "Anti", "R-out", "R-in"};
        for (std::size_t i = 0; i < kHoldButtons.size(); ++i) {
            if (i > 0) {
                ImGui::SameLine();
            }
            if (ImGui::SmallButton(kHoldButtons.at(i))) {
                action = static_cast<HudAction>(static_cast<std::size_t>(HudAction::kHoldOff) + i);
            }
        }
        ImGui::TextDisabled("Shift/Ctrl throttle  Z/X full/cut  W/S A/D Q/E rotate");
        ImGui::TextDisabled("T damping  R hold  L PD/LQR  ,/. warp  (>100x: rails, no hold)");
        ImGui::Text("%-14s %s", "Environment", f.j2 ? "point mass + J2" : "two-body");
        if (ImGui::Button("Restart: two-body")) {
            action = HudAction::kRestartTwoBody;
        }
        ImGui::SameLine();
        if (ImGui::Button("Restart: with J2")) {
            action = HudAction::kRestartJ2;
        }
        if (ImGui::Button("Verify replay")) {
            action = HudAction::kVerifyReplay;
        }
        ImGui::SameLine();
        ImGui::TextWrapped("%s", f.replay_status[0] != '\0'
                                     ? f.replay_status
                                     : "re-run this session from its input log");
    }

    if (ImGui::CollapsingHeader("Orbit", ImGuiTreeNodeFlags_DefaultOpen)) {
        text_km("Altitude", tm.altitude_m);
        ImGui::Text("%-14s %.3f m/s", "Speed", tm.speed_mps);
        if (!tm.elements_valid) {
            ImGui::TextDisabled("No conic elements (near-parabolic or radial trajectory)");
        }
        text_km("Periapsis", tm.periapsis_altitude_m);
        text_km("Apoapsis", tm.apoapsis_altitude_m);
        if (std::isnan(tm.period_s)) {
            ImGui::Text("%-14s n/a", "Period");
        } else if (std::isinf(tm.period_s)) {
            ImGui::Text("%-14s none (escape)", "Period");
        } else {
            ImGui::Text("%-14s %.2f min", "Period", tm.period_s / 60.0);
        }
        ImGui::TextDisabled(f.j2 ? "Osculating elements (instantaneous; wobble under J2)"
                                 : "Osculating elements (instantaneous two-body orbit)");
        ImGui::Text("%-14s %.3f km", "a", tm.elements.a_m / 1000.0);
        ImGui::Text("%-14s %.6f", "e", tm.elements.e);
        ImGui::Text("%-14s %.4f deg", "i", tm.elements.i_rad * kRadToDeg);
        ImGui::Text("%-14s %.4f deg", "RAAN", wrap_deg(tm.elements.raan_rad));
        ImGui::Text("%-14s %.4f deg", "Arg periapsis", wrap_deg(tm.elements.argp_rad));
        ImGui::Text("%-14s %.4f deg", "True anomaly", wrap_deg(tm.elements.nu_rad));
    }

    if (ImGui::CollapsingHeader("Vehicle", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Text("%-14s %.1f kg", "Mass", tm.mass_kg);
        const double capacity = cfg.mass.capacity_kg();
        std::array<char, 32> fuel{};
        std::snprintf(fuel.data(), fuel.size(), "%.1f / %.0f kg", tm.propellant_kg, capacity);
        ImGui::ProgressBar(static_cast<float>(tm.propellant_kg / capacity), ImVec2(-1.0F, 0.0F),
                           fuel.data());
        ImGui::Text("%-14s %.1f m/s", "Delta-v left", tm.delta_v_remaining_mps);
        ImGui::TextDisabled("main engine, rocket equation v_e ln(m / m_dry)");
    }

    if (ImGui::CollapsingHeader("Attitude", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Text("%-14s %.2f deg", "Engine to pro.", tm.thrust_axis_to_prograde_rad * kRadToDeg);
        ImGui::TextDisabled("angle from body +x to the velocity (0 = prograde)");
        ImGui::Text("%-14s %+.3f %+.3f %+.3f deg/s", "Body rates", tm.w_body.x() * kRadToDeg,
                    tm.w_body.y() * kRadToDeg, tm.w_body.z() * kRadToDeg);
    }

    if (!cfg.wheels.empty() && ImGui::CollapsingHeader("Wheels", ImGuiTreeNodeFlags_DefaultOpen)) {
        for (std::size_t j = 0; j < cfg.wheels.size(); ++j) {
            const double sat = tm.wheel_saturation[j];
            std::array<char, 48> label{};
            std::snprintf(label.data(), label.size(), "%+.3f N m s  (%+.0f%%)", tm.wheel_h_nms[j],
                          100.0 * sat);
            ImGui::Text("Wheel %zu", j + 1);
            ImGui::SameLine();
            ImGui::ProgressBar(static_cast<float>(std::abs(sat)), ImVec2(-1.0F, 0.0F),
                               label.data());
        }
        ImGui::TextDisabled("bar: wheel speed relative to the body / limit");
    }

    ImGui::Separator();
    ImGui::Text("Camera %.3g km  |  tick %lld", f.camera_distance_m / 1000.0,
                static_cast<long long>(f.tick));
    ImGui::TextDisabled("drag: turn view   scroll: zoom");
    ImGui::Text("simcore %.*s  |  %d FPS%s", static_cast<int>(simcore::version().size()),
                simcore::version().data(), f.fps, f.fell_behind ? "  (behind)" : "");
    ImGui::End();
    return action;
}

}  // namespace simapp
