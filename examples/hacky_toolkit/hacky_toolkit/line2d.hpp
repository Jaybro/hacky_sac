#pragma once

#include <random>
#include <vector>

#include "plane.hpp"

using Line2d = Plane2d;
using Line2f = Plane2f;

std::vector<Eigen::Vector2d> GenerateDataset(
    Line2d const& line, double p_outlier, std::size_t* p_n_inliers) {
  // TODO: Expose later.
  std::size_t n_points = 500;
  double length = 100.0 / 2.0;
  double sigma = 0.2;
  double outlier = sigma * 100.0;

  std::default_random_engine generator;
  std::normal_distribution<double> distribution_normal(0.0, sigma);
  std::uniform_real_distribution<double> distribution_uniform(0.0, 1.0);
  auto sampler_normal = [&]() -> double {
    return distribution_normal(generator);
  };
  auto sampler_uniform = [&]() -> double {
    return distribution_uniform(generator);
  };

  // Distance from the origin
  double distance = line.d();
  Eigen::Vector2d n = line.normal();
  // Line as if the normal was derived by the right hand rule
  Eigen::Vector2d v(n.y(), -n.x());
  // Offset applied to shift the line from the origin
  Eigen::Vector2d d = n * distance;
  std::vector<Eigen::Vector2d> dataset(n_points);

  // Line centered around d.
  auto generate_point_on_line = [&]() -> Eigen::Vector2d {
    double sign = (sampler_uniform() < 0.5) ? 1.0 : -1.0;
    return d + v * sampler_uniform() * length * sign;
  };

  // Gives a random point along l with noise in the normal direction
  auto generate_noisy = [&]() -> Eigen::Vector2d {
    return generate_point_on_line() + n * sampler_normal();
  };

  auto generate_outlier = [&]() -> Eigen::Vector2d {
    double sign = (sampler_uniform() < 0.5) ? 1.0 : -1.0;
    return generate_point_on_line() + n * sampler_uniform() * outlier * sign;
  };

  std::size_t n_inliers = 0;

  for (std::size_t i = 0; i < n_points; ++i) {
    if (sampler_uniform() > p_outlier) {
      dataset[i] = generate_noisy();
      n_inliers++;
    } else {
      dataset[i] = generate_outlier();
    }
  }

  *p_n_inliers = n_inliers;

  return dataset;
}
