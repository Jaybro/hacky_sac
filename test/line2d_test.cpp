#include <gtest/gtest.h>

#include <hacky_toolkit/line2d.hpp>

TEST(RansacTest, Model) {
  Eigen::Vector2d a(0.0, 0.0);
  Eigen::Vector2d b(4.0, 4.0);
  Eigen::Vector3d l = CalculateLine(a, b);
  Eigen::Vector2d c0(2.0, 2.0);
  Eigen::Vector2d c1(0.0, 4.0);
  Eigen::Vector2d c2(4.0, 0.0);

  EXPECT_TRUE(TestPointOnLine(l, c0, std::numeric_limits<double>::epsilon()));
  EXPECT_FALSE(TestPointOnLine(l, c1, std::numeric_limits<double>::epsilon()));
  EXPECT_FALSE(TestPointOnLine(l, c2, std::numeric_limits<double>::epsilon()));
}
