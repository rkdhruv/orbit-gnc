#pragma once

#include <optional>

#include "simcore/dynamics/state.hpp"
#include "simcore/sim/simulation.hpp"

namespace simapp {

// The 3D view: Earth, the orbit line and the spacecraft with its attitude, seen by a camera that
// orbits the spacecraft. Everything is drawn relative to the spacecraft (the floating origin) in
// two passes, because no single depth range covers a 2 m spacecraft and a 6,378 km Earth:
//   far pass:  1 unit = 1 km  (Earth, orbit line, a marker for the spacecraft when zoomed out)
//   near pass: 1 unit = 1 m   (the spacecraft), after clearing the depth buffer
// The camera direction is fixed in inertial (render) axes; drag to turn it, scroll to zoom.
class Scene {
public:
    // mouse_free: false while ImGui is using the mouse, so dragging a panel doesn't turn the view.
    void handle_input(bool mouse_free);
    // predicted: the state just after a planned burn (drawn as a second orbit line, with a marker
    // at the burn point), if a maneuver node is set.
    void draw(const simcore::Snapshot& snap, const std::optional<simcore::State>& predicted) const;
    [[nodiscard]] double camera_distance_m() const { return distance_m_; }

private:
    float yaw_rad_{0.6F};
    float pitch_rad_{0.3F};
    double distance_m_{25.0};
};

}  // namespace simapp
