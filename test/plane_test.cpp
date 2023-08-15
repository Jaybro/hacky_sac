#include <gtest/gtest.h>

#include <hacky_toolkit/plane_utils.hpp>

using namespace hacky_toolkit;

TEST(PlaneTest, Random) {
  double max_distance = 100.0;
  auto plane = RandomHyperplane<double, 2>(max_distance);
  auto point = RandomPointOnHyperplane(plane, max_distance);

  // Random plane.
  EXPECT_LE(std::abs(plane.offset()), max_distance);
  EXPECT_DOUBLE_EQ(1.0, plane.normal().squaredNorm());

  // Random point on plane.
  EXPECT_TRUE(TestPointOnHyperplane(plane, point));
  EXPECT_DOUBLE_EQ(
      max_distance, (point - ProjectionOriginOnHyperplane(plane)).norm());
}

TEST(PlaneTest, Tests) {
  Eigen::Vector2d a(0.0, 0.0);
  Eigen::Vector2d b(4.0, 4.0);

  Eigen::Hyperplane<double, 2> plane =
      Eigen::Hyperplane<double, 2>::Through(a, b);

  Eigen::Vector2d c0(2.0, 2.0);
  Eigen::Vector2d c1(0.0, 4.0);
  Eigen::Vector2d c2(4.0, 0.0);

  EXPECT_TRUE(TestPointOnHyperplane(plane, c0));
  EXPECT_FALSE(TestPointOnHyperplane(plane, c1));
  EXPECT_FALSE(TestPointOnHyperplane(plane, c2));
}
