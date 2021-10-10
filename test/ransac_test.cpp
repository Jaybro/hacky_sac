#include <gtest/gtest.h>

#include <hacky_sac/ransac.hpp>

using namespace hacky_sac;

TEST(RansacTest, FixedIterationAdaptor) {
  std::size_t iterations = 100;
  FixedIterationAdaptor adaptor(iterations);

  EXPECT_EQ(iterations, adaptor.Iterations());
  EXPECT_EQ(iterations, adaptor.Iterations(0));
}

TEST(RansacTest, ProbabilisticIterationAdaptor) {
  std::size_t n_samples = 3;
  std::size_t n_datums = 1000;
  double p_inlier_a_priori = 0.4;

  hacky_sac::ProbabilisticIterationAdaptor adaptor(
      0.999, p_inlier_a_priori, n_samples, n_datums, 0);

  EXPECT_LT(adaptor.Iterations(n_datums / 2), adaptor.Iterations());
}
