/**
BSD 3-Clause License

This file is part of the Basalt project.
https://gitlab.com/VladyslavUsenko/basalt-headers.git

Copyright (c) 2019, Vladyslav Usenko and Nikolaus Demmel.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

@file
@brief Implementation of equirectangular (360x180) camera model
*/

#pragma once

#include <basalt/camera/camera_static_assert.hpp>

#include <basalt/utils/sophus_utils.hpp>

namespace basalt {

using std::atan2;
using std::cos;
using std::sin;
using std::sqrt;

/// @brief Equirectangular (360x180) camera model
///
/// Models a full-sphere projection (e.g. Insta360 stitched output). The
/// 4 stored parameters \f$ \mathbf{i} = [f_x, f_y, c_x, c_y]^T \f$ are
/// derived from image resolution and held **frozen** by the optimizer:
///   - \f$ f_x = W / (2\pi),\ f_y = H / \pi \f$
///   - \f$ c_x = W / 2,\     c_y = H / 2 \f$
///
/// `setFromInit` and `operator+=` are intentional no-ops so the LM update
/// in basalt's calibration loop leaves the projection unchanged. To
/// construct from a resolution use \ref fromResolution.
///
/// Projection follows the standard equirectangular convention with axes
/// X right, Y down, Z forward (basalt's standard):
///   - longitude \f$ \lambda = \text{atan2}(X, Z) \in [-\pi, \pi] \f$
///   - latitude  \f$ \phi = \text{asin}(Y / r) \in [-\pi/2, \pi/2] \f$
///   - \f$ u = f_x \lambda + c_x,\ v = f_y \phi + c_y \f$
template <typename Scalar_ = double>
class EquirectangularCamera {
 public:
  using Scalar = Scalar_;
  static constexpr int N = 4;  ///< Number of intrinsic parameters.

  /// @brief Pixel residuals wrap in u with period W = 2π·fx.
  ///
  /// Equirectangular projection has a branch cut at longitude = ±π, where
  /// the pixel column jumps between u=0 and u=W. AprilGrid corners
  /// straddling that seam produce reprojection residuals of ~W pixels even
  /// when the geometric error is tiny; the optimizer must wrap the u
  /// residual by `W * round(Δu/W)` before accumulating gradients. The
  /// LinearizeBase trait `camera_has_azimuthal_wrap` opts equi in.
  static constexpr bool kAzimuthalWrap = true;

  using Vec2 = Eigen::Matrix<Scalar, 2, 1>;
  using Vec4 = Eigen::Matrix<Scalar, 4, 1>;

  using VecN = Eigen::Matrix<Scalar, N, 1>;

  using Mat24 = Eigen::Matrix<Scalar, 2, 4>;
  using Mat2N = Eigen::Matrix<Scalar, 2, N>;

  using Mat42 = Eigen::Matrix<Scalar, 4, 2>;
  using Mat4N = Eigen::Matrix<Scalar, 4, N>;

  /// @brief Default constructor with zero intrinsics.
  EquirectangularCamera() { param_.setZero(); }

  /// @brief Construct camera model with given vector of intrinsics.
  ///
  /// @param[in] p vector of intrinsic parameters [fx, fy, cx, cy] —
  ///              typically the output of \ref fromResolution(W, H).
  explicit EquirectangularCamera(const VecN& p) { param_ = p; }

  /// @brief Build the frozen-intrinsics vector from an image resolution.
  ///
  /// @param[in] W image width  in pixels
  /// @param[in] H image height in pixels
  static EquirectangularCamera fromResolution(int W, int H) {
    VecN p;
    const Scalar pi = Sophus::Constants<Scalar>::pi();
    p[0] = Scalar(W) / (Scalar(2) * pi);
    p[1] = Scalar(H) / pi;
    p[2] = Scalar(W) / Scalar(2);
    p[3] = Scalar(H) / Scalar(2);
    return EquirectangularCamera(p);
  }

  /// @brief Cast to different scalar type.
  template <class Scalar2>
  EquirectangularCamera<Scalar2> cast() const {
    return EquirectangularCamera<Scalar2>(param_.template cast<Scalar2>());
  }

  /// @brief Camera model name.
  ///
  /// @return "equi"
  static std::string getName() { return "equi"; }

  /// @brief Project a 3D point onto the equirectangular image plane.
  ///
  /// Projection is well-defined for any direction except the two points
  /// on the y-axis (\f$X = Z = 0\f$) and the origin (\f$\|p\| = 0\f$).
  /// Those return `false`; the output `proj` is undefined in that case.
  ///
  /// @param[in]  p3d            3D point (or 4D homogeneous; w ignored)
  /// @param[out] proj           2D pixel location
  /// @param[out] d_proj_d_p3d   optional 2×{3,4} Jacobian wrt p3d
  /// @param[out] d_proj_d_param optional 2×N Jacobian wrt intrinsics
  ///                            (zero — intrinsics are frozen)
  template <class DerivedPoint3D, class DerivedPoint2D,
            class DerivedJ3D = std::nullptr_t,
            class DerivedJparam = std::nullptr_t>
  inline bool project(const Eigen::MatrixBase<DerivedPoint3D>& p3d,
                      Eigen::MatrixBase<DerivedPoint2D>& proj,
                      DerivedJ3D d_proj_d_p3d = nullptr,
                      DerivedJparam d_proj_d_param = nullptr) const {
    checkProjectionDerivedTypes<DerivedPoint3D, DerivedPoint2D, DerivedJ3D,
                                DerivedJparam, N>();

    const typename EvalOrReference<DerivedPoint3D>::Type p3d_eval(p3d);

    const Scalar& fx = param_[0];
    const Scalar& fy = param_[1];
    const Scalar& cx = param_[2];
    const Scalar& cy = param_[3];

    const Scalar& x = p3d_eval[0];
    const Scalar& y = p3d_eval[1];
    const Scalar& z = p3d_eval[2];

    const Scalar s = x * x + z * z;  // squared horizontal radius
    const Scalar r2 = s + y * y;     // squared 3D norm
    const Scalar eps = Sophus::Constants<Scalar>::epsilonSqrt();
    const bool is_valid = (s > eps) && (r2 > eps);

    const Scalar r = sqrt(r2);
    const Scalar sqrt_s = sqrt(s);
    const Scalar lon = atan2(x, z);
    const Scalar lat = (r > eps) ? std::asin(y / r) : Scalar(0);

    proj[0] = fx * lon + cx;
    proj[1] = fy * lat + cy;

    if constexpr (!std::is_same_v<DerivedJ3D, std::nullptr_t>) {
      BASALT_ASSERT(d_proj_d_p3d);
      d_proj_d_p3d->setZero();
      if (is_valid) {
        // ∂lon/∂x =  z/s,  ∂lon/∂y = 0,  ∂lon/∂z = -x/s
        (*d_proj_d_p3d)(0, 0) = fx * z / s;
        (*d_proj_d_p3d)(0, 1) = Scalar(0);
        (*d_proj_d_p3d)(0, 2) = -fx * x / s;
        // ∂lat/∂x = -x*y / (r² · √s)
        // ∂lat/∂y =  √s  /  r²
        // ∂lat/∂z = -y*z / (r² · √s)
        const Scalar inv_r2 = Scalar(1) / r2;
        const Scalar inv_r2_sqrts = inv_r2 / sqrt_s;
        (*d_proj_d_p3d)(1, 0) = -fy * x * y * inv_r2_sqrts;
        (*d_proj_d_p3d)(1, 1) = fy * sqrt_s * inv_r2;
        (*d_proj_d_p3d)(1, 2) = -fy * y * z * inv_r2_sqrts;
      }
    } else {
      UNUSED(d_proj_d_p3d);
    }

    if constexpr (!std::is_same_v<DerivedJparam, std::nullptr_t>) {
      BASALT_ASSERT(d_proj_d_param);
      // Intrinsics are frozen → ∂proj/∂param = 0.
      d_proj_d_param->setZero();
    } else {
      UNUSED(d_proj_d_param);
    }

    return is_valid;
  }

  /// @brief Unproject a pixel to a 3D unit ray.
  ///
  /// Always succeeds for finite pixel coordinates. The returned 3D point
  /// (or 4D point with w=0) lies on the unit sphere.
  ///
  /// @param[in]  proj            2D pixel location
  /// @param[out] p3d             3D unit ray (or 4D with w=0)
  /// @param[out] d_p3d_d_proj    optional {3,4}×2 Jacobian wrt proj
  /// @param[out] d_p3d_d_param   optional {3,4}×N Jacobian wrt intrinsics
  ///                             (zero — intrinsics are frozen)
  template <class DerivedPoint2D, class DerivedPoint3D,
            class DerivedJ2D = std::nullptr_t,
            class DerivedJparam = std::nullptr_t>
  inline bool unproject(const Eigen::MatrixBase<DerivedPoint2D>& proj,
                        Eigen::MatrixBase<DerivedPoint3D>& p3d,
                        DerivedJ2D d_p3d_d_proj = nullptr,
                        DerivedJparam d_p3d_d_param = nullptr) const {
    checkUnprojectionDerivedTypes<DerivedPoint2D, DerivedPoint3D, DerivedJ2D,
                                  DerivedJparam, N>();

    const typename EvalOrReference<DerivedPoint2D>::Type proj_eval(proj);

    const Scalar& fx = param_[0];
    const Scalar& fy = param_[1];
    const Scalar& cx = param_[2];
    const Scalar& cy = param_[3];

    const Scalar lon = (proj_eval[0] - cx) / fx;
    const Scalar lat = (proj_eval[1] - cy) / fy;

    const Scalar cl = cos(lat);
    const Scalar sl = sin(lat);
    const Scalar cL = cos(lon);
    const Scalar sL = sin(lon);

    p3d.setZero();
    p3d[0] = cl * sL;
    p3d[1] = sl;
    p3d[2] = cl * cL;

    if constexpr (!std::is_same_v<DerivedJ2D, std::nullptr_t>) {
      BASALT_ASSERT(d_p3d_d_proj);
      d_p3d_d_proj->setZero();
      // d_X_d_u = cos(lat)·cos(lon)/fx;  d_X_d_v = -sin(lat)·sin(lon)/fy
      // d_Y_d_u = 0;                      d_Y_d_v = cos(lat)/fy
      // d_Z_d_u = -cos(lat)·sin(lon)/fx;  d_Z_d_v = -sin(lat)·cos(lon)/fy
      (*d_p3d_d_proj)(0, 0) = cl * cL / fx;
      (*d_p3d_d_proj)(0, 1) = -sl * sL / fy;
      (*d_p3d_d_proj)(1, 0) = Scalar(0);
      (*d_p3d_d_proj)(1, 1) = cl / fy;
      (*d_p3d_d_proj)(2, 0) = -cl * sL / fx;
      (*d_p3d_d_proj)(2, 1) = -sl * cL / fy;
    } else {
      UNUSED(d_p3d_d_proj);
    }

    if constexpr (!std::is_same_v<DerivedJparam, std::nullptr_t>) {
      BASALT_ASSERT(d_p3d_d_param);
      // Intrinsics are frozen → ∂p3d/∂param = 0.
      d_p3d_d_param->setZero();
    } else {
      UNUSED(d_p3d_d_param);
    }

    return true;
  }

  /// @brief No-op: intrinsics are frozen, derived from resolution at
  /// construction. Calibrator passes a Vec4 init that we ignore — see
  /// `cam_calib.cpp` for the equirect-specific init path.
  inline void setFromInit(const Vec4& /*init*/) {}

  /// @brief No-op: intrinsics are frozen during LM updates.
  void operator+=(const VecN& /*inc*/) {}

  /// @brief Returns a const reference to the (frozen) intrinsics
  /// \f$ [f_x, f_y, c_x, c_y]^T \f$.
  const VecN& getParam() const { return param_; }

  /// @brief Test camera instances (calibrated for common Insta360 stitched
  /// resolutions).
  static Eigen::aligned_vector<EquirectangularCamera> getTestProjections() {
    Eigen::aligned_vector<EquirectangularCamera> res;
    res.emplace_back(fromResolution(3840, 1920));  // 4K equirect
    res.emplace_back(fromResolution(5760, 2880));  // Insta360 X4 stitched
    return res;
  }

  /// @brief Test resolutions matching \ref getTestProjections.
  static Eigen::aligned_vector<Eigen::Vector2i> getTestResolutions() {
    Eigen::aligned_vector<Eigen::Vector2i> res;
    res.emplace_back(3840, 1920);
    res.emplace_back(5760, 2880);
    return res;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 private:
  VecN param_;  // [fx, fy, cx, cy], frozen.
};

}  // namespace basalt
