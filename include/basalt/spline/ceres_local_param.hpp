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
@brief Generic local parametrization for Sophus Lie group types to be used with
ceres.
*/

/**
File adapted from Sophus

Copyright 2011-2017 Hauke Strasdat
          2012-2017 Steven Lovegrove

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to
deal in the Software without restriction, including without limitation the
rights  to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.
*/

#pragma once

#if __has_include(<ceres/local_parameterization.h>)
#define BASALT_HAS_CERES_LOCAL_PARAMETERIZATION 1
#include <ceres/local_parameterization.h>
#else
#define BASALT_HAS_CERES_LOCAL_PARAMETERIZATION 0
#include <ceres/manifold.h>
#endif
#include <sophus/se3.hpp>

namespace basalt {

/// @brief Local parametrization for ceres that can be used with Sophus Lie
/// group implementations.
template <class Groupd>
class LieLocalParameterization
#if BASALT_HAS_CERES_LOCAL_PARAMETERIZATION
    : public ceres::LocalParameterization {
#else
    : public ceres::Manifold {
#endif
 public:
  virtual ~LieLocalParameterization() {}

  using Tangentd = typename Groupd::Tangent;

  /// @brief plus operation for Ceres
  ///
  ///  T * exp(x)
  ///
  virtual bool Plus(double const* T_raw, double const* delta_raw,
                    double* T_plus_delta_raw) const {
    Eigen::Map<Groupd const> const T(T_raw);
    Eigen::Map<Tangentd const> const delta(delta_raw);
    Eigen::Map<Groupd> T_plus_delta(T_plus_delta_raw);
    T_plus_delta = T * Groupd::exp(delta);
    return true;
  }

  ///@brief Jacobian of plus operation for Ceres
  ///
  /// Dx T * exp(x)  with  x=0
  ///
  virtual bool ComputeJacobian(double const* T_raw,
                               double* jacobian_raw) const {
    Eigen::Map<Groupd const> T(T_raw);
    Eigen::Map<Eigen::Matrix<double, Groupd::num_parameters, Groupd::DoF,
                             Eigen::RowMajor>>
        jacobian(jacobian_raw);
    jacobian = T.Dx_this_mul_exp_x_at_0();
    return true;
  }

#if BASALT_HAS_CERES_LOCAL_PARAMETERIZATION
  ///@brief Global size
  virtual int GlobalSize() const { return Groupd::num_parameters; }

  ///@brief Local size
  virtual int LocalSize() const { return Groupd::DoF; }
#else
  virtual bool PlusJacobian(double const* T_raw, double* jacobian_raw) const {
    return ComputeJacobian(T_raw, jacobian_raw);
  }

  virtual bool Minus(double const* y_raw, double const* x_raw,
                     double* y_minus_x_raw) const {
    Eigen::Map<Groupd const> const x(x_raw);
    Eigen::Map<Groupd const> const y(y_raw);
    Eigen::Map<Tangentd> y_minus_x(y_minus_x_raw);
    y_minus_x = (x.inverse() * y).log();
    return true;
  }

  virtual bool MinusJacobian(double const* T_raw, double* jacobian_raw) const {
    Eigen::Map<Groupd const> T(T_raw);
    Eigen::Matrix<double, Groupd::num_parameters, Groupd::DoF> J_plus =
        T.Dx_this_mul_exp_x_at_0();
    Eigen::Map<Eigen::Matrix<double, Groupd::DoF, Groupd::num_parameters,
                             Eigen::RowMajor>>
        J_minus(jacobian_raw);
    J_minus = (J_plus.transpose() * J_plus).ldlt().solve(J_plus.transpose());
    return true;
  }

  virtual int AmbientSize() const { return Groupd::num_parameters; }

  virtual int TangentSize() const { return Groupd::DoF; }
#endif
};

}  // namespace basalt
