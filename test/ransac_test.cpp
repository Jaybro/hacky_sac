#include <gtest/gtest.h>

#include <hacky_sac/ransac.hpp>
#include <hacky_toolkit/plane_utils.hpp>

using namespace hacky_sac;
using namespace hacky_toolkit;

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

TEST(RansacTest, EstimateModel) {
  std::size_t n_inliers_ground_truth;
  Plane3d original = Plane3d::Random();
  std::size_t n_points = 100;
  double p_inlier_ground_truth = 0.95;
  double p_inlier_a_priori = 0.2;
  double inlier_noise_sigma = 0.2;
  double inlier_max_radius = 10.0;

  auto dataset = GenerateDataset(
      original,
      n_points,
      p_inlier_ground_truth,
      inlier_noise_sigma,
      inlier_max_radius,
      &n_inliers_ground_truth);

  auto f_model_estimator =
      [&dataset](std::vector<std::size_t> const& samples) -> Plane3d {
    return Plane3d(
        dataset[samples[0]], dataset[samples[1]], dataset[samples[2]]);
  };

  double inlier_threshold = inlier_noise_sigma * 2.0;

  auto f_sample_tester = [&dataset, &inlier_threshold](
                             Plane3d const& model, std::size_t index) -> bool {
    return model.Test(dataset[index], inlier_threshold);
  };

  std::size_t n_samples = 3;
  hacky_sac::ProbabilisticIterationAdaptor adapter_probabilistic(
      0.999, p_inlier_a_priori, n_samples, dataset.size(), 0);

  auto result = hacky_sac::EstimateModel<Plane3d>(
      n_samples,
      dataset.size(),
      adapter_probabilistic,
      f_model_estimator,
      f_sample_tester);

  EXPECT_LT(result.n_iterations, adapter_probabilistic.Iterations());

  EXPECT_GT(
      static_cast<double>(result.n_inliers) /
          static_cast<double>(dataset.size()),
      0.5);
}
