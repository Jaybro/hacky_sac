#pragma once

#include <Eigen/Dense>

//! \brief A Plane (or hyperplane) is a susbspace of which its dimension is one
//! less than of its ambient space. The template argument of the dimension of
//! the plane equals that of its ambient space. I.e., a Plane<float, 2> is a
//! line in a 2d space. A plane is represented by a vector of Dim + 1 parameters
//! that can be used to obtain the general form of the plane equation:
//! normal.dot(x) + d = 0
template <typename Scalar_, std::size_t Dim_>
class Plane {
 private:
  using VectorType = Eigen::Matrix<Scalar_, Dim_ + 1, 1>;

 public:
  inline Plane() = default;

  template <typename Derived>
  inline Plane(Eigen::MatrixBase<Derived> const& p) : a_(p) {}

  template <typename Derived>
  inline Plane(Eigen::MatrixBase<Derived> const& normal, Scalar_ d) {
    a_.template head<Dim_>() = normal;
    a_(Dim_) = d;
  }

  //! \brief Creates a normalized plane using points. The spatial dimension must
  //! equal 2 for this constructor.
  template <typename Derived0, typename Derived1>
  inline Plane(
      Eigen::MatrixBase<Derived0> const& x,
      Eigen::MatrixBase<Derived1> const& y) {
    static_assert(Dim_ == 2);
    static_assert(
        Derived0::SizeAtCompileTime == Dim_ ||
        Derived0::SizeAtCompileTime == Dim_ + 1);
    static_assert(Derived0::SizeAtCompileTime == Derived1::SizeAtCompileTime);
    if constexpr (Derived0::SizeAtCompileTime == Dim_) {
      a_ = x.homogeneous().cross(y.homogeneous());
    } else {
      a_ = x.cross(y);
    }
    Normalize();
  }

  template <typename Derived>
  inline Scalar_ SignedDistance(Eigen::MatrixBase<Derived> const& x) const {
    static_assert(
        Derived::SizeAtCompileTime == Dim_ ||
        Derived::SizeAtCompileTime == Dim_ + 1);
    if constexpr (Derived::SizeAtCompileTime == Dim_) {
      return a_.dot(x.homogeneous());
    } else {
      return a_.dot(x);
    }
  }

  template <typename Derived>
  inline Scalar_ Distance(Eigen::MatrixBase<Derived> const& x) const {
    return std::abs(SignedDistance(x));
  }

  template <typename Derived>
  inline bool Positive(Eigen::MatrixBase<Derived> const& x) const {
    return SignedDistance(x) > Scalar_(0.0);
  }

  template <typename Derived>
  inline bool Negative(Eigen::MatrixBase<Derived> const& x) const {
    return SignedDistance(x) < Scalar_(0.0);
  }

  template <typename Derived>
  inline bool Test(
      Eigen::MatrixBase<Derived> const& x,
      Scalar_ threshold = std::numeric_limits<Scalar_>::epsilon()) const {
    return Distance(x) < threshold;
  }

  //! \brief Normalizes the plane such that the length of the normal (the first
  //! Dim parameters) equals 1.
  inline void Normalize() { a_ /= a_.template head<Dim_>().norm(); }

  //! \brief Point on the plane closest to the origin.
  inline Eigen::Matrix<Scalar_, Dim_, 1> Origin() const {
    return normal() * -d();
  }

  inline auto normal() const {
    // Eigen::VectorBlock<VectorType const, Dim_>
    return a_.template head<Dim_>();
  }

  //! \brief Negative of the distance from the origin to the plane in the
  //! direction of the normal. That is, the closest point on the plane to the
  //! origin equals: plane.normal() * -plane.d()
  Scalar_ const& d() const {
    // Last element.
    return a_(Dim_);
  }

  auto const& a() const {
    // VectorType
    return a_;
  }

  //! \brief Generates a Plane with a random normal at a distance somewhere in
  //! the range of [-max_distance:max_distance].
  static Plane Random(Scalar_ max_distance = Scalar_(1.0)) {
    return {
        Eigen::Matrix<Scalar_, Dim_, 1>::Random().normalized(),
        Eigen::Matrix<Scalar_, 1, 1>::Random()(0) * max_distance};
  }

 private:
  VectorType a_;
};

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

using Plane2d = Plane<double, 2>;
using Plane3d = Plane<double, 3>;

using Plane2f = Plane<float, 2>;
using Plane3f = Plane<float, 3>;
