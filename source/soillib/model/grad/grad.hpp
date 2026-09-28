#pragma once

#include <soillib/soillib.hpp>

#include <silt/core/types.hpp>
#include <silt/core/shape.hpp>
#include <silt/core/tensor.hpp>

namespace soil {

//
// Gradient Operators -- three ways to turn a height field into a
// 2-component (dz/dx, dz/dy) field. All three feed the same downstream
// soil::slope (magnitude) and soil::normal (normalize trick) below.
//

//! Gradient, Linear Descending (Steepest) Upwind.
//! The single steepest-descent D4 neighbor per cell -- only the gradient
//! component along that neighbor's axis is nonzero, the other is 0. This
//! is the old graph-based steepest-descent slope, decoupled from any
//! precomputed flow graph. Prone to directional artifacts; prefer
//! grad_quadratic or grad_centered for a smoother erosion surface.
silt::tensor_t<float> grad_steepest(const silt::tensor_t<float>& tensor, const silt::vec3 scale);

//! Gradient, Quadratic Descending Upwind (Godunov/eikonal upwind).
//! Per axis, the steeper of the two one-sided downhill differences, zero
//! if both neighbors are uphill.
silt::tensor_t<float> grad_quadratic(const silt::tensor_t<float>& tensor, const silt::vec3 scale);

//! Gradient, Centered Finite Differences (5-point stencil).
//! Falls back to a one-sided difference at the boundary.
silt::tensor_t<float> grad_centered(const silt::tensor_t<float>& tensor, const silt::vec3 scale);

//! Slope Magnitude from a precomputed 2-component gradient field
//! (sqrt(gx^2 + gy^2)) -- pairs with any of the three grad_* above.
silt::tensor_t<float> slope(const silt::tensor_t<float>& gradient);

//! 2D Tensor Laplacian
silt::tensor_t<float> laplacian(const silt::tensor_t<float>& tensor, const silt::vec2 scale);

}
