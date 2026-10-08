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
@brief Implementation of the MEI (unified omnidirectional + radial-tangential)
camera model
*/

#pragma once

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>

#include <basalt/camera/camera_static_assert.hpp>

#include <basalt/utils/sophus_utils.hpp>

namespace basalt {

using std::sqrt;

/// Process-wide options for MeiCamera, read once from the environment.
///
/// - BASALT_MEI_XI_INIT=<xi0>: xi that setFromInit seeds (default 1).
/// - BASALT_MEI_XI_FIXED=1: keep xi out of the optimisation. The xi column of
///   the projection Jacobian w.r.t. the intrinsics is zeroed, so the
///   optimiser's step for xi is exactly 0 (its LM damping is diagonal-scaled
///   with a floor, the xi row decouples).
/// - BASALT_MEI_DIST_INIT="k1 k2 p1 p2 [k3 k4]": distortion that setFromInit
///   seeds. Needed with xi > 1: undistorted MEI only lifts normalised radii
///   r^2 <= 1/(xi^2 - 1), which on a ~190 deg lens excludes the image rim.
struct MeiCameraOptions {
  static constexpr double kDefaultXiInit = 1.0;

  static double xiInit() {
    static const double v = parseXiInit(std::getenv("BASALT_MEI_XI_INIT"));
    return v;
  }

  /// Parses a BASALT_MEI_XI_INIT value. Unset or empty gives the default;
  /// anything that is not a single finite number >= 0 warns and gives the
  /// default, since a silent xi0 = 0 seeds a pinhole at half the focal length.
  static double parseXiInit(const char* s) {
    if (s == nullptr || *s == '\0') return kDefaultXiInit;
    char* end = nullptr;
    const double xi = std::strtod(s, &end);
    if (end == s || *end != '\0' || !std::isfinite(xi) || xi < 0.0) {
      std::cerr << "BASALT_MEI_XI_INIT=\"" << s
                << "\" is not a finite number >= 0; using xi0 = "
                << kDefaultXiInit << ".\n";
      return kDefaultXiInit;
    }
    return xi;
  }
  static bool xiFixed() {
    static const bool v = [] {
      const char* s = std::getenv("BASALT_MEI_XI_FIXED");
      return s != nullptr && *s != '\0' && std::strcmp(s, "0") != 0;
    }();
    return v;
  }
  static std::vector<double> distInit() {
    std::vector<double> v;
    const char* s = std::getenv("BASALT_MEI_DIST_INIT");
    if (s != nullptr) {
      std::istringstream is(s);
      double d;
      while (is >> d) v.push_back(d);
    }
    return v;
  }
};

/// @brief MEI camera model: unified (sphere) projection followed by
/// radial-tangential distortion (Mei & Rives 2007). This is the per-unit lens
/// model Insta360 stores in its factory calibration.
///
/// NK = 2 ("mei"): N=9 parameters
/// \f$ \left[f_x, f_y, c_x, c_y, \xi, k_1, k_2, p_1, p_2 \right]^T \f$.
/// NK = 4 ("mei4"): N=11 parameters, radial polynomial to r^8,
/// \f$ \left[f_x, f_y, c_x, c_y, \xi, k_1, k_2, p_1, p_2, k_3, k_4
/// \right]^T \f$ (the first nine as in "mei").
/// See \ref project and \ref unproject functions for more details.
template <typename Scalar_ = double, int NK = 2>
class MeiCamera {
  static_assert(NK == 2 || NK == 4, "MeiCamera supports 2 or 4 radial terms");

 public:
  using Scalar = Scalar_;
  static constexpr int N = 9 + (NK - 2);  ///< Number of intrinsic parameters.

  using Vec2 = Eigen::Matrix<Scalar, 2, 1>;
  using Vec4 = Eigen::Matrix<Scalar, 4, 1>;

  using VecN = Eigen::Matrix<Scalar, N, 1>;

  using Mat22 = Eigen::Matrix<Scalar, 2, 2>;
  using Mat24 = Eigen::Matrix<Scalar, 2, 4>;
  using Mat2N = Eigen::Matrix<Scalar, 2, N>;

  using Mat42 = Eigen::Matrix<Scalar, 4, 2>;
  using Mat4N = Eigen::Matrix<Scalar, 4, N>;

  /// @brief Default constructor with zero intrinsics
  MeiCamera() { param_.setZero(); }

  /// @brief Construct camera model with given vector of intrinsics
  ///
  /// @param[in] p vector of intrinsic parameters, see class description
  explicit MeiCamera(const VecN& p) { param_ = p; }

  /// @brief Cast to different scalar type
  template <class Scalar2>
  MeiCamera<Scalar2, NK> cast() const {
    return MeiCamera<Scalar2, NK>(param_.template cast<Scalar2>());
  }

  /// @brief Camera model name
  ///
  /// @return "mei" or "mei4"
  static std::string getName() { return NK == 2 ? "mei" : "mei4"; }

  /// @brief Project the point and optionally compute Jacobians
  ///
  /// \f{align}{
  ///   m &= \frac{1}{z + \xi d} \begin{bmatrix} x \\ y \end{bmatrix},
  ///   \quad d = \sqrt{x^2 + y^2 + z^2},
  ///   \\ \pi(\mathbf{x}, \mathbf{i}) &=
  ///   \begin{bmatrix} f_x m'_x + c_x \\ f_y m'_y + c_y \end{bmatrix},
  ///   \quad m' = \mathrm{distort}(m)
  /// \f}
  /// with radial-tangential distortion
  /// \f$ m'_x = m_x (1 + k_1 r^2 + k_2 r^4 [+ k_3 r^6 + k_4 r^8])
  ///   + 2 p_1 m_x m_y + p_2 (r^2 + 2 m_x^2) \f$,
  /// \f$ m'_y = m_y (\ldots) + p_1 (r^2 + 2 m_y^2) + 2 p_2 m_x m_y \f$.
  ///
  /// Valid for \f$ z > -d / \xi \f$ if \f$ \xi > 1 \f$, else
  /// \f$ z > -\xi d \f$.
  ///
  /// @param[in] p3d point to project
  /// @param[out] proj result of projection
  /// @param[out] d_proj_d_p3d if not nullptr computed Jacobian of projection
  /// with respect to p3d
  /// @param[out] d_proj_d_param point if not nullptr computed Jacobian of
  /// projection with respect to intrinsic parameters
  /// @return if projection is valid
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
    const Scalar& xi = param_[4];

    const Scalar& x = p3d_eval[0];
    const Scalar& y = p3d_eval[1];
    const Scalar& z = p3d_eval[2];

    const Scalar d = sqrt(x * x + y * y + z * z);
    const Scalar den = z + xi * d;

    const bool is_valid =
        d > Sophus::Constants<Scalar>::epsilon() &&
        den > Sophus::Constants<Scalar>::epsilon() &&
        (xi > Scalar(1) ? z * xi > -d : z > -xi * d);
    // Keep the arithmetic finite for invalid points; the result is discarded.
    const Scalar den_safe =
        den > Sophus::Constants<Scalar>::epsilon() ? den : Scalar(1);

    const Scalar mx = x / den_safe;
    const Scalar my = y / den_safe;

    Vec2 md;
    Mat22 d_md_d_m;
    Scalar r2, rad;
    distort(mx, my, md, &d_md_d_m, r2, rad);

    proj = Vec2(fx * md[0] + cx, fy * md[1] + cy);

    if constexpr (!std::is_same_v<DerivedJ3D, std::nullptr_t> ||
                  !std::is_same_v<DerivedJparam, std::nullptr_t>) {
      // d m / d [x y z] and d m / d xi
      const Scalar inv_den = Scalar(1) / den_safe;
      const Scalar d_safe =
          d > Sophus::Constants<Scalar>::epsilon() ? d : Scalar(1);
      Eigen::Matrix<Scalar, 2, 3> d_m_d_p;
      const Scalar dden_dx = xi * x / d_safe;
      const Scalar dden_dy = xi * y / d_safe;
      const Scalar dden_dz = Scalar(1) + xi * z / d_safe;
      d_m_d_p(0, 0) = inv_den - mx * dden_dx * inv_den;
      d_m_d_p(0, 1) = -mx * dden_dy * inv_den;
      d_m_d_p(0, 2) = -mx * dden_dz * inv_den;
      d_m_d_p(1, 0) = -my * dden_dx * inv_den;
      d_m_d_p(1, 1) = inv_den - my * dden_dy * inv_den;
      d_m_d_p(1, 2) = -my * dden_dz * inv_den;

      Mat22 F;
      F << fx, Scalar(0), Scalar(0), fy;
      const Mat22 F_Jd = F * d_md_d_m;

      if constexpr (!std::is_same_v<DerivedJ3D, std::nullptr_t>) {
        BASALT_ASSERT(d_proj_d_p3d);
        d_proj_d_p3d->setZero();
        d_proj_d_p3d->template block<2, 3>(0, 0) = F_Jd * d_m_d_p;
      } else {
        UNUSED(d_proj_d_p3d);
      }

      if constexpr (!std::is_same_v<DerivedJparam, std::nullptr_t>) {
        BASALT_ASSERT(d_proj_d_param);
        d_proj_d_param->setZero();
        (*d_proj_d_param)(0, 0) = md[0];
        (*d_proj_d_param)(1, 1) = md[1];
        (*d_proj_d_param)(0, 2) = Scalar(1);
        (*d_proj_d_param)(1, 3) = Scalar(1);

        if (!MeiCameraOptions::xiFixed()) {
          // d m / d xi = -m * d / den
          const Vec2 d_m_d_xi(-mx * d * inv_den, -my * d * inv_den);
          d_proj_d_param->col(4) = F_Jd * d_m_d_xi;
        }

        const Scalar r4 = r2 * r2;
        (*d_proj_d_param)(0, 5) = fx * mx * r2;
        (*d_proj_d_param)(1, 5) = fy * my * r2;
        (*d_proj_d_param)(0, 6) = fx * mx * r4;
        (*d_proj_d_param)(1, 6) = fy * my * r4;
        (*d_proj_d_param)(0, 7) = fx * Scalar(2) * mx * my;
        (*d_proj_d_param)(1, 7) = fy * (r2 + Scalar(2) * my * my);
        (*d_proj_d_param)(0, 8) = fx * (r2 + Scalar(2) * mx * mx);
        (*d_proj_d_param)(1, 8) = fy * Scalar(2) * mx * my;
        if constexpr (NK == 4) {
          const Scalar r6 = r4 * r2;
          const Scalar r8 = r4 * r4;
          (*d_proj_d_param)(0, 9) = fx * mx * r6;
          (*d_proj_d_param)(1, 9) = fy * my * r6;
          (*d_proj_d_param)(0, 10) = fx * mx * r8;
          (*d_proj_d_param)(1, 10) = fy * my * r8;
        }
      } else {
        UNUSED(d_proj_d_param);
      }
    } else {
      UNUSED(d_proj_d_p3d);
      UNUSED(d_proj_d_param);
    }

    return is_valid;
  }

  /// @brief Unproject the point
  ///
  /// The distortion is inverted iteratively (Gauss-Newton on distort), then
  /// the undistorted point is lifted to the unit sphere:
  /// \f{align}{
  ///  \pi^{-1}(\mathbf{u}, \mathbf{i}) &=
  ///  \frac{\xi + \sqrt{1 + (1 - \xi^2) r^2}}{1 + r^2}
  ///  \begin{bmatrix} m_x \\ m_y \\ 1 \end{bmatrix} -
  ///  \begin{bmatrix} 0 \\ 0 \\ \xi \end{bmatrix}.
  /// \f}
  /// Valid if \f$ 1 + (1 - \xi^2) r^2 \ge 0 \f$ and the undistortion
  /// converged.
  ///
  /// Jacobians of the unprojection are not implemented (as for
  /// PinholeRadtan8Camera); nothing in basalt requests them at runtime.
  ///
  /// @param[in] proj point to unproject
  /// @param[out] p3d result of unprojection
  /// @param[out] d_p3d_d_proj not supported, must be nullptr
  /// @param[out] d_p3d_d_param not supported, must be nullptr
  /// @return if unprojection is valid
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
    const Scalar& xi = param_[4];

    const Vec2 md((proj_eval[0] - cx) / fx, (proj_eval[1] - cy) / fy);

    // Undistort: Gauss-Newton on distort(m) = md, starting from md.
    Vec2 m = md;
    bool converged = false;
    for (int it = 0; it < 30; ++it) {
      Vec2 mdi;
      Mat22 J;
      Scalar r2_unused, rad_unused;
      distort(m[0], m[1], mdi, &J, r2_unused, rad_unused);
      const Vec2 e = md - mdi;
      if (e.squaredNorm() <
          Sophus::Constants<Scalar>::epsilon() *
              Sophus::Constants<Scalar>::epsilon()) {
        converged = true;
        break;
      }
      m += J.inverse() * e;
      if (!m.allFinite()) break;
    }
    if (!converged) {
      Vec2 mdi;
      Scalar r2_unused, rad_unused;
      distort<std::nullptr_t>(m[0], m[1], mdi, nullptr, r2_unused, rad_unused);
      converged = m.allFinite() &&
                  (md - mdi).norm() < Sophus::Constants<Scalar>::epsilonSqrt() *
                                          Scalar(1e-2);
    }

    const Scalar r2 = m.squaredNorm();
    const Scalar s = Scalar(1) + (Scalar(1) - xi * xi) * r2;
    const bool is_valid = converged && s >= Scalar(0);

    const Scalar k = (xi + sqrt(s > Scalar(0) ? s : Scalar(0))) / (Scalar(1) + r2);

    p3d.setZero();
    p3d[0] = k * m[0];
    p3d[1] = k * m[1];
    p3d[2] = k - xi;

    if constexpr (!std::is_same_v<DerivedJ2D, std::nullptr_t>) {
      BASALT_ASSERT(false);  // Not implemented
      UNUSED(d_p3d_d_proj);
    } else {
      UNUSED(d_p3d_d_proj);
    }
    if constexpr (!std::is_same_v<DerivedJparam, std::nullptr_t>) {
      BASALT_ASSERT(false);  // Not implemented
      UNUSED(d_p3d_d_param);
    } else {
      UNUSED(d_p3d_d_param);
    }

    return is_valid;
  }

  /// @brief Set parameters from initialization
  ///
  /// basalt's line-fit initialisation returns [fx, fy, cx, cy] in the unified
  /// (alpha = 0.5, i.e. MEI xi = 1) convention, whose near-axis focal is
  /// fx. MEI keeps that near-axis focal with f = fx (1 + xi0), xi0 from
  /// BASALT_MEI_XI_INIT (default 1). Distortion from BASALT_MEI_DIST_INIT, else
  /// zero.
  ///
  /// @param[in] init vector [fx, fy, cx, cy]
  inline void setFromInit(const Vec4& init) {
    const Scalar xi0 = Scalar(MeiCameraOptions::xiInit());
    param_.setZero();
    param_[0] = init[0] * (Scalar(1) + xi0);
    param_[1] = init[1] * (Scalar(1) + xi0);
    param_[2] = init[2];
    param_[3] = init[3];
    param_[4] = xi0;
    const std::vector<double> d0 = MeiCameraOptions::distInit();
    if (d0.size() == static_cast<size_t>(N - 5)) {
      for (int i = 0; i < N - 5; ++i) param_[5 + i] = Scalar(d0[i]);
    } else if (!d0.empty()) {
      std::cerr << "BASALT_MEI_DIST_INIT has " << d0.size() << " values, "
                << getName() << " has " << N - 5
                << " distortion parameters; distortion seed ignored.\n";
    }
  }

  /// @brief Increment intrinsic parameters by inc and clamp xi to xi >= 0
  ///
  /// @param[in] inc increment vector
  void operator+=(const VecN& inc) {
    param_ += inc;
    param_[4] = std::max(param_[4], Scalar(0));
  }

  /// @brief Returns a const reference to the intrinsic parameters vector
  ///
  /// The order is [fx, fy, cx, cy, xi, k1, k2, p1, p2 (, k3, k4)].
  /// @return const reference to the intrinsic parameters vector
  const VecN& getParam() const { return param_; }

  /// @brief Projections used for unit-tests
  static Eigen::aligned_vector<MeiCamera> getTestProjections() {
    Eigen::aligned_vector<MeiCamera> res;
    VecN vec1;

    // Insta360 X6 lens 0, factory record (3008x3008), small tangential terms
    if constexpr (NK == 2) {
      vec1 << 2828.58, 2828.58, 1498.01, 1506.10, 2.45543, 1.30284, -0.99313,
          -0.0009, 0.0007;
    } else {
      vec1 << 2828.58, 2828.58, 1498.01, 1506.10, 2.45543, 1.30284, -0.99313,
          -0.0009, 0.0007, 2.55915, 9.18316;
    }
    res.emplace_back(vec1);

    // Moderate fisheye, xi < 1
    if constexpr (NK == 2) {
      vec1 << 400, 405, 320, 240, 0.8, -0.2, 0.05, 0.001, -0.0005;
    } else {
      vec1 << 400, 405, 320, 240, 0.8, -0.2, 0.05, 0.001, -0.0005, 0.01,
          -0.002;
    }
    res.emplace_back(vec1);

    return res;
  }

  /// @brief Resolutions used for unit-tests
  static Eigen::aligned_vector<Eigen::Vector2i> getTestResolutions() {
    Eigen::aligned_vector<Eigen::Vector2i> res;
    res.emplace_back(3008, 3008);
    res.emplace_back(640, 480);
    return res;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 private:
  /// Radial-tangential distortion of the normalised point (mx, my); also
  /// returns r^2, the radial factor - 1, and optionally d(md)/d(m).
  template <class DerivedJ = Mat22*>
  inline void distort(const Scalar mx, const Scalar my, Vec2& md, DerivedJ J,
                      Scalar& r2, Scalar& rad) const {
    const Scalar& k1 = param_[5];
    const Scalar& k2 = param_[6];
    const Scalar& p1 = param_[7];
    const Scalar& p2 = param_[8];
    Scalar k3(0), k4(0);
    if constexpr (NK == 4) {
      k3 = param_[9];
      k4 = param_[10];
    }

    r2 = mx * mx + my * my;
    rad = r2 * (k1 + r2 * (k2 + r2 * (k3 + r2 * k4)));
    const Scalar mxy = mx * my;
    md[0] = mx * (Scalar(1) + rad) + Scalar(2) * p1 * mxy +
            p2 * (r2 + Scalar(2) * mx * mx);
    md[1] = my * (Scalar(1) + rad) + p1 * (r2 + Scalar(2) * my * my) +
            Scalar(2) * p2 * mxy;

    if constexpr (!std::is_same_v<DerivedJ, std::nullptr_t>) {
      // d(rad)/d(r2)
      const Scalar drad =
          k1 + r2 * (Scalar(2) * k2 +
                     r2 * (Scalar(3) * k3 + r2 * Scalar(4) * k4));
      (*J)(0, 0) = Scalar(1) + rad + Scalar(2) * mx * mx * drad +
                   Scalar(2) * p1 * my + Scalar(6) * p2 * mx;
      (*J)(0, 1) = Scalar(2) * mxy * drad + Scalar(2) * p1 * mx +
                   Scalar(2) * p2 * my;
      (*J)(1, 0) = (*J)(0, 1);
      (*J)(1, 1) = Scalar(1) + rad + Scalar(2) * my * my * drad +
                   Scalar(6) * p1 * my + Scalar(2) * p2 * mx;
    } else {
      UNUSED(J);
    }
  }

  VecN param_;
};

}  // namespace basalt
