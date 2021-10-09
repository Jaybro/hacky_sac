#pragma once

#include <random>
#include <vector>

#include "plane.hpp"

namespace hacky_toolkit {

//! \brief Esimates a Plane from Dim_+1 or more points.
template <typename Scalar_, int Dim_>
Plane<Scalar_, Dim_> EstimatePlane(
    std::vector<Eigen::Matrix<Scalar_, Dim_, 1>> const& points,
    std::vector<bool> const& mask) {
  using MatrixType = Eigen::Matrix<Scalar_, Dim_ + 1, Dim_ + 1>;

  MatrixType A = MatrixType::Zero();

  for (std::size_t i = 0; i < points.size(); ++i) {
    if (mask[i]) {
      // TODO Can't seem to do this at once
      Eigen::Matrix<Scalar_, Dim_ + 1, 1> v = points[i].homogeneous();
      A += v * v.transpose();
    }
  }

  // For square matrices NoQRPreconditioner is the most optimal.
  // https://eigen.tuxfamily.org/dox/classEigen_1_1JacobiSVD.html
  Eigen::JacobiSVD<MatrixType, Eigen::NoQRPreconditioner> d(
      A, Eigen::ComputeFullV);

  Plane<Scalar_, Dim_> plane(d.matrixV().col(Dim_));
  plane.Normalize();
  return plane;
}

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

  // Generates a random point on the disc around plane.Origin() with noise in
  // the normal direction.
  auto generate_noisy = [&]() -> PointType {
    Scalar_ noise = sampler_normal();
    return plane.RandomPoint(inlier_max_radius * sampler_uniform()) + n * noise;
  };

  // The same as generate_noisy, but then points are (expected to be) farther
  // from the plane.
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

}  // namespace hacky_toolkit
