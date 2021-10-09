#pragma once

#include <random>
#include <vector>

#include "plane.hpp"

using Line2d = Plane2d;
using Line2f = Plane2f;

template <typename Scalar_, int Dim_>
std::vector<Eigen::Matrix<Scalar_, Dim_, 1>> GenerateDataset(
    Plane<Scalar_, Dim_> const& plane,
    std::size_t n_points,
    Scalar_ p_inlier,
    Scalar_ inlier_noise_sigma,
    Scalar_ inlier_max_radius,
    std::size_t* p_n_inliers) {
  using PointType = Eigen::Matrix<Scalar_, Dim_, 1>;

  Scalar_ max_outlier_distance = inlier_noise_sigma * Scalar_(100.0);

  std::default_random_engine generator;
  std::normal_distribution<Scalar_> distribution_normal(
      Scalar_(0.0), inlier_noise_sigma);
  std::uniform_real_distribution<Scalar_> distribution_uniform;
  auto sampler_normal = [&]() -> Scalar_ {
    return distribution_normal(generator);
  };
  auto sampler_uniform = [&]() -> Scalar_ {
    return distribution_uniform(generator);
  };

  PointType n = plane.normal();
  std::vector<PointType> dataset(n_points);

  // Gives a random point along l with noise in the normal direction
  auto generate_noisy = [&]() -> PointType {
    Scalar_ noise = sampler_normal();
    return plane.RandomPoint(inlier_max_radius * sampler_uniform()) + n * noise;
  };

  auto generate_outlier = [&]() -> PointType {
    Scalar_ sign =
        (sampler_uniform() < Scalar_(0.5)) ? Scalar_(1.0) : Scalar_(-1.0);
    return plane.RandomPoint(inlier_max_radius * sampler_uniform()) +
           n * sampler_uniform() * max_outlier_distance * sign;
  };

  std::size_t n_inliers = 0;

  for (std::size_t i = 0; i < n_points; ++i) {
    if (sampler_uniform() < p_inlier) {
      dataset[i] = generate_noisy();
      n_inliers++;
    } else {
      dataset[i] = generate_outlier();
    }
  }

  *p_n_inliers = n_inliers;

  return dataset;
}
