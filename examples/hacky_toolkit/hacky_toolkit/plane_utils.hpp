#pragma once

#include <Eigen/Geometry>
#include <random>
#include <vector>

namespace hacky_toolkit {

//! \brief Generates a Plane with a random normal at a distance somewhere in
//! the range of [-max_distance:max_distance].
template <typename Scalar_, int Dim_>
inline Eigen::Hyperplane<Scalar_, Dim_> RandomHyperplane(
    Scalar_ max_distance = Scalar_(1.0)) {
  return {
      Eigen::Matrix<Scalar_, Dim_, 1>::Random().normalized(),
      Eigen::Matrix<Scalar_, 1, 1>::Random()(0) * max_distance};
}

//! \brief Tests is a point is considered to lie on the plane.
//! \returns true if \p x is on the plane.
template <typename Scalar_, int Dim_, typename Derived_>
inline bool TestPointOnHyperplane(
    Eigen::Hyperplane<Scalar_, Dim_> const& plane,
    Eigen::MatrixBase<Derived_> const& x,
    Scalar_ threshold = Eigen::NumTraits<Scalar_>::dummy_precision()) {
  return plane.absDistance(x) < threshold;
}

//! \brief Point on the plane closest to the origin.
template <typename Scalar_, int Dim_>
inline Eigen::Matrix<Scalar_, Dim_, 1> ProjectionOriginOnHyperplane(
    Eigen::Hyperplane<Scalar_, Dim_> const& plane) {
  return plane.normal() * -plane.offset();
}

//! \brief Generates a random point on the unit hypersphere of the subspace
//! defined by this plane at a distance of \p distance from the point returned
//! by the Origin() method. The plane does not have to be normalized.
//! \returns A random point on the plane.
template <typename Scalar_, int Dim_>
inline Eigen::Matrix<Scalar_, Dim_, 1> RandomPointOnHyperplane(
    Eigen::Hyperplane<Scalar_, Dim_> const& plane,
    Scalar_ distance = Scalar_(1.0)) {
  // Perturb the normal with some noise.
  // TODO: What are the odds that the added noise equals 0?
  Eigen::Matrix<Scalar_, Dim_, 1> v =
      plane.normal() + Eigen::Matrix<Scalar_, Dim_, 1>::Random();
  // Apply Gram-Schmidt process to get an orthogonal vector. Ie, subtract the
  // projection of v on the normal from v.
  // TODO: When the plane is guaranteed to be normalized, we don't need the
  // devision by the squared length of the normal.
  return (v -
          plane.normal() * plane.normal().dot(v) / plane.normal().squaredNorm())
                 .normalized() *
             distance +
         ProjectionOriginOnHyperplane(plane);
}

//! \brief Least squares plane estimation using Dim_+1 or more input points..
//! \returns A plane that fits the input points.
template <typename Scalar_, int Dim_>
Eigen::Hyperplane<Scalar_, Dim_> EstimatePlane(
    std::vector<Eigen::Matrix<Scalar_, Dim_, 1>> const& points,
    std::vector<bool> const& mask) {
  static_assert(Dim_ != Eigen::Dynamic);
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

  Eigen::Hyperplane<Scalar_, Dim_> plane;
  plane.coeffs() = d.matrixV().col(Dim_);
  plane.normalize();
  return plane;
}

template <typename Scalar_, int Dim_>
std::vector<Eigen::Matrix<Scalar_, Dim_, 1>> GenerateDataset(
    Eigen::Hyperplane<Scalar_, Dim_> const& plane,
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
    return RandomPointOnHyperplane(
               plane, inlier_max_radius * sampler_uniform()) +
           n * noise;
  };

  // The same as generate_noisy, but then points are (expected to be) farther
  // from the plane.
  auto generate_outlier = [&]() -> PointType {
    Scalar_ sign =
        (sampler_uniform() < Scalar_(0.5)) ? Scalar_(1.0) : Scalar_(-1.0);
    return RandomPointOnHyperplane(
               plane, inlier_max_radius * sampler_uniform()) +
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
