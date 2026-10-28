// M1.1: State — plain translational state (epoch, ECI position, ECI velocity).

#include <gtest/gtest.h>

#include <type_traits>

#include <Eigen/Dense>

#include "simcore/dynamics/state.hpp"
#include "simcore/orbits/epoch.hpp"

using simcore::Epoch;
using simcore::State;

TEST(State, DefaultIsAtRestAtOriginAtJ2000) {
    const State s{};
    EXPECT_EQ(s.t, Epoch::j2000());
    EXPECT_EQ(s.r_eci, Eigen::Vector3d::Zero());
    EXPECT_EQ(s.v_eci, Eigen::Vector3d::Zero());
}

TEST(State, AggregateInitialisationAndCopyPreserveValues) {
    // A 420 km circular-orbit state (values from test_point_mass_gravity.cpp), one minute after
    // J2000.
    const State s{Epoch::j2000() + 60.0, Eigen::Vector3d(6'798'137.0, 0.0, 0.0),
                  Eigen::Vector3d(0.0, 7657.27, 0.0)};
    const State copy = s;  // value semantics: integrators will copy states freely

    EXPECT_EQ(copy.t, Epoch::from_tt_seconds_since_j2000(60));
    EXPECT_EQ(copy.r_eci, Eigen::Vector3d(6'798'137.0, 0.0, 0.0));
    EXPECT_EQ(copy.v_eci, Eigen::Vector3d(0.0, 7657.27, 0.0));
    static_assert(std::is_copy_constructible_v<State> && std::is_copy_assignable_v<State>);
}
