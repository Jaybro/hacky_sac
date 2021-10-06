#pragma once

#include <Eigen/Dense>

//! \brief This class represents a plane (or hyperplane). The dimension argument
//! equals that of its ambient space. I.e., a Plane<float, 2> is a line in a 2d
//! space. A plane is represented by a vector of Dim + 1 parameters that can be
//! used to obtain the general form of the plane equation: normal.dot(x) + d = 0
template <typename Scalar_, std::size_t Dim_>
class Plane {
 private:
  using VectorType = Eigen::Matrix<Scalar_, Dim_ + 1, 1>;

 public:
  template <typename Derived>
  inline Plane(Eigen::MatrixBase<Derived> const& p) {
    a_ = p;
    Normalize();
  }

  template <typename Derived>
  inline Plane(Eigen::MatrixBase<Derived> const& a, Scalar_ d) {
    a_.template head<Dim_>() = a;
    a_(Dim_) = d;
    Normalize();
  }

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

  auto normal() const {
    // Eigen::VectorBlock<VectorType const, Dim_>
    return a_.template head<Dim_>();
  }

  Scalar_ const& d() const {
    // Last element.
    return a_(Dim_);
  }

  auto const& a() const {
    // VectorType
    return a_;
  }

 private:
  inline void Normalize() { a_ /= a_.template head<Dim_>().norm(); }

  VectorType a_;
};

using Plane2d = Plane<double, 2>;
using Plane3d = Plane<double, 3>;

using Plane2f = Plane<float, 2>;
using Plane3f = Plane<float, 3>;
