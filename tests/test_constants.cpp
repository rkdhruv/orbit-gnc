// M1.1: physical constants match the values fixed in CLAUDE.md.
//
// Every downstream reference check (420 km orbit, Hohmann, rocket equation) is derived from
// these numbers, so a typo here would silently shift every result. Exact equality is correct:
// the literals in the header and here parse to the same double.

#include <gtest/gtest.h>

#include "simcore/constants.hpp"

TEST(Constants, EarthValuesMatchProjectSpec) {
    EXPECT_EQ(simcore::earth::mu, 3.986004418e14);  // m^3/s^2 (EGM96 / WGS-84 GM)
    EXPECT_EQ(simcore::earth::radius, 6.378137e6);  // m (WGS-84 equatorial radius)
    EXPECT_EQ(simcore::earth::j2, 1.08263e-3);      // dimensionless
}

TEST(Constants, StandardGravityMatchesProjectSpec) {
    EXPECT_EQ(simcore::g0, 9.80665);  // m/s^2, exact by definition (CGPM 1901)
}
