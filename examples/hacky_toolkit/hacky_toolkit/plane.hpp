#pragma once

#include <Eigen/Dense>

//! \brief This class represents a plane (or hyperplane). The dimension argument
//! equals that of its ambient space. I.e., a Plane<float, 2> is a line in a 2d
//! space.
template <typename Scalar_, std::size_t Dim_>
class Plane {
 public:
  template <typename Derived>
  inline Plane(Eigen::MatrixBase<Derived> const& p) {
    p_ = p;
    Normalize();
  }

  template <typename Derived>
  inline Plane(Eigen::MatrixBase<Derived> const& v, Scalar_ d) {
    p_.template head<Dim_>() = v;
    p_(Dim_) = d;
    Normalize();
  }

  template <typename Derived0, typename Derived1>
  inline Plane(
      Eigen::MatrixBase<Derived0> const& x,
      Eigen::MatrixBase<Derived1> const& y) {
    static_assert(
        Derived0::SizeAtCompileTime == Dim_ ||
        Derived0::SizeAtCompileTime == Dim_ + 1);
    static_assert(Derived0::SizeAtCompileTime == Derived1::SizeAtCompileTime);
    if constexpr (Derived0::SizeAtCompileTime == Dim_) {
      p_ = x.homogeneous().cross(y.homogeneous());
    } else {
      p_ = x.cross(y);
    }
    Normalize();
  }

  template <typename Derived>
  inline Scalar_ SignedDistance(Eigen::MatrixBase<Derived> const& x) const {
    static_assert(
        Derived::SizeAtCompileTime == Dim_ ||
        Derived::SizeAtCompileTime == Dim_ + 1);
    if constexpr (Derived::SizeAtCompileTime == Dim_) {
      return p_.dot(x.homogeneous());
    } else {
      return p_.dot(x);
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

 private:
  inline void Normalize() { p_ /= p_.template head<Dim_>().norm(); }

  Eigen::Matrix<Scalar_, Dim_ + 1, 1> p_;
};

using Plane2d = Plane<double, 2>;
using Plane3d = Plane<double, 3>;

using Plane2f = Plane<float, 2>;
using Plane3f = Plane<float, 3>;
