#pragma once

#include <random>
#include <vector>

#include "plane.hpp"

using Line2d = Plane2d;
using Line2f = Plane2f;

Eigen::Vector3d CalculateLine(
    Eigen::Vector2d const& a, Eigen::Vector2d const& b) {
  Eigen::Vector2d v = (b - a).normalized();
  Eigen::Vector2d n(-v.y(), v.x());
  double d = n.dot(a);

  return {n.x(), n.y(), -d};

  // Eigen::Vector3d l = a.homogeneous().cross(b.homogeneous());
  // l /= l.head<2>().norm();
  // return l;
}

bool TestPointOnLine(
    Eigen::Vector3d const& l, Eigen::Vector2d const& p, double threshold) {
  return std::abs(l.dot(p.homogeneous())) < threshold;
}

std::vector<Eigen::Vector2d> GenerateDataset(
    double p_outlier, std::size_t* p_n_inliers, Eigen::Vector3d* p_line) {
  std::default_random_engine generator;
  std::normal_distribution<double> distribution_normal(0.0, 0.2);
  std::uniform_real_distribution<double> distribution_uniform(0.0, 1.0);
  auto sampler_normal = [&]() -> double {
    return distribution_normal(generator);
  };
  auto sampler_uniform = [&]() -> double {
    return distribution_uniform(generator);
  };

  double length = 100.0;
  // Distance from the origin
  double distance = 44.0;
  // Line direction
  Eigen::Vector2d v(1.0, 0.65);
  v.normalize();
  // Normal of the line by the right hand rule
  Eigen::Vector2d n(-v.y(), v.x());
  // Offset applied to shift the line from the origin
  Eigen::Vector2d d = n * distance;
  std::size_t n_points = 500;
  std::vector<Eigen::Vector2d> dataset(n_points);

  // Gives a random point along l with noise in the normal direction
  auto generate_noisy = [&]() -> Eigen::Vector2d {
    return v * length * sampler_uniform() + n * sampler_normal() + d;
  };

  auto generate_outlier = [&]() -> Eigen::Vector2d {
    double sign = (sampler_uniform() < 0.5) ? 1.0 : -1.0;

    return v * length * sampler_uniform() +
           n * length * sampler_uniform() * sign + d;
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
  *p_line = {n.x(), n.y(), -distance};

  return dataset;
}
