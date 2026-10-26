// M0 smoke test: proves simcore, Eigen and GoogleTest are wired together.

#include <gtest/gtest.h>

#include <Eigen/Dense>

#include "simcore/version.hpp"

TEST(Smoke, VersionIsNotEmpty) {
    EXPECT_FALSE(simcore::version().empty());
}

TEST(Smoke, EigenIsUsable) {
    // 3-4-5 triangle: sqrt(9 + 16) = 5 is exact in IEEE double, so EXPECT_EQ is fine.
    const Eigen::Vector3d v(3.0, 4.0, 0.0);
    EXPECT_EQ(v.norm(), 5.0);
}
