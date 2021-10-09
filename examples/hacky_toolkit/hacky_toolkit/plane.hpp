#pragma once

#include <Eigen/Dense>

namespace hacky_toolkit {

//! \brief A Plane (or hyperplane) is a susbspace of which its dimension is one
//! less than of its ambient space. The template argument of the dimension of
//! the plane equals that of its ambient space. I.e., a Plane<float, 2> is a
//! line in a 2d space. A plane is represented by a vector of Dim + 1 parameters
//! that can be used to obtain the general form of the plane equation:
//! a.dot((x, 1)) + d = 0
template <typename Scalar_, int Dim_>
class Plane {
  static_assert(Dim_ > 0, "DYNAMIC_DIMENSION_NOT_SUPPORTED");

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

  //! \brief Returns true if \p x is considered to lie on the plane.
  template <typename Derived>
  inline bool Test(
      Eigen::MatrixBase<Derived> const& x,
      Scalar_ threshold = Eigen::NumTraits<Scalar_>::dummy_precision()) const {
    return Distance(x) < threshold;
  }

  //! \brief Normalizes the plane such that the length of the normal (the first
  //! Dim parameters of a()) equals 1.
  inline void Normalize() { a_ /= a_.template head<Dim_>().norm(); }

  //! \brief Point on the plane closest to the origin.
  inline Eigen::Matrix<Scalar_, Dim_, 1> Origin() const {
    return normal() * -d();
  }

  //! \brief Generates a random point on the unit hypersphere of the subspace
  //! defined by this plane at a distance of \p distance from the point returned
  //! by the Origin() method. The plane does not have to be normalized.
  inline Eigen::Matrix<Scalar_, Dim_, 1> RandomPoint(
      Scalar_ distance = Scalar_(1.0)) const {
    // Perturb the normal with some noise.
    // What are the odds that the added noise equals 0?
    Eigen::Matrix<Scalar_, Dim_, 1> v =
        normal() + Eigen::Matrix<Scalar_, Dim_, 1>::Random();
    // Apply Gram-Schmidt process to get an orthogonal vector. Ie, subtract the
    // projection of v on the normal from v.
    return (v - normal() * normal().dot(v) / normal().squaredNorm())
                   .normalized() *
               distance +
           Origin();
  }

  //! \brief Plane of the normal.
  inline auto normal() const {
    // Eigen::VectorBlock<VectorType const, Dim_>
    return a_.template head<Dim_>();
  }

  //! \brief Negative of the distance from the origin to the plane in the
  //! direction of the normal. That is, the closest point on the plane to the
  //! origin equals: plane.normal() * -plane.d()
  Scalar_ const& d() const { return a_(Dim_); }

  //! \brief The parameters of the plane. They describe the General form of the
  //! plane equation: a.dot((x, 1)) = 0
  auto const& a() const { return a_; }

  //! \brief Generates a Plane with a random normal at a distance somewhere in
  //! the range of [-max_distance:max_distance].
  static Plane Random(Scalar_ max_distance = Scalar_(1.0)) {
    return {
        Eigen::Matrix<Scalar_, Dim_, 1>::Random().normalized(),
        Eigen::Matrix<Scalar_, 1, 1>::Random()(0) * max_distance};
  }

 private:
  Eigen::Matrix<Scalar_, Dim_ + 1, 1> a_;
};

using Plane2d = Plane<double, 2>;
using Plane3d = Plane<double, 3>;

using Plane2f = Plane<float, 2>;
using Plane3f = Plane<float, 3>;

}  // namespace hacky_toolkit
