#include <gtest/gtest.h>

#include <hacky_toolkit/plane.hpp>

TEST(PlaneTest, Construction) {
  Eigen::Vector2d a(0.0, 1.0);
  Eigen::Vector2d b(4.0, 1.0);

  {
    Plane2d plane(a, b);

    EXPECT_TRUE(plane.normal().isApprox(Eigen::Vector2d::UnitY()));
    EXPECT_DOUBLE_EQ(1.0, -plane.d());
  }

  {
    Plane2d plane(b, a);

    EXPECT_TRUE(plane.normal().isApprox(Eigen::Vector2d::UnitY() * -1.0));
    EXPECT_DOUBLE_EQ(1.0, plane.d());
  }
}

TEST(PlaneTest, Tests) {
  Eigen::Vector2d a(0.0, 0.0);
  Eigen::Vector2d b(4.0, 4.0);

  Plane2d plane(a, b);

  Eigen::Vector2d c0(2.0, 2.0);
  Eigen::Vector2d c1(0.0, 4.0);
  Eigen::Vector2d c2(4.0, 0.0);

  EXPECT_TRUE(plane.Test(c0));
  EXPECT_FALSE(plane.Test(c1));
  EXPECT_FALSE(plane.Test(c2));
  EXPECT_TRUE(plane.Positive(c1));
  EXPECT_TRUE(plane.Negative(c2));
}
