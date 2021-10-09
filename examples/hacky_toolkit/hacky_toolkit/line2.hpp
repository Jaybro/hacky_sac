#pragma once

#include <random>
#include <vector>

#include "plane.hpp"

using Line2d = Plane2d;
using Line2f = Plane2f;

std::vector<Eigen::Vector2d> GenerateDataset(
    Line2d const& line,
    std::size_t n_points,
    double p_inlier,
    double inlier_noise_sigma,
    std::size_t* p_n_inliers) {
  // TODO: Expose later.
  double length = 100.0 / 2.0;
  double max_outlier_distance = inlier_noise_sigma * 100.0;

  std::default_random_engine generator;
  std::normal_distribution<double> distribution_normal(0.0, inlier_noise_sigma);
  std::uniform_real_distribution<double> distribution_uniform(0.0, 1.0);
  auto sampler_normal = [&]() -> double {
    return distribution_normal(generator);
  };
  auto sampler_uniform = [&]() -> double {
    return distribution_uniform(generator);
  };

  Eigen::Vector2d n = line.normal();
  std::vector<Eigen::Vector2d> dataset(n_points);

  // Gives a random point along l with noise in the normal direction
  auto generate_noisy = [&]() -> Eigen::Vector2d {
    double noise = sampler_normal();
    return line.RandomPoint(length) + n * noise;
  };

  auto generate_outlier = [&]() -> Eigen::Vector2d {
    double sign = (sampler_uniform() < 0.5) ? 1.0 : -1.0;
    return line.RandomPoint(length) +
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
