#ifndef SOILLIB_OP_EROSION_THERMAL_CU
#define SOILLIB_OP_EROSION_THERMAL_CU
#define HAS_CUDA

#include <soillib/core/types.hpp>
#include <soillib/core/buffer.hpp>
#include <soillib/core/index.hpp>

#include <soillib/op/erosion.hpp>

//! Thermal Erosion Algorithms
//!
//! These device functions implement various methods to compute
//! the time-differential of a height-map from thermal erosion.
//! 
//! \todo Research alternative thermal erosion implementations

namespace soil {

//! Slope Limiting Mass Wasting
__device__ void slope_limit(const buffer_t<float>& height, buffer_t<float>& height_diff, const flat_t<2>& index, const ivec2 ipos, const param_t param) {

  if(index.oob(ipos))
    return;

  // Get Non-Out-of-Bounds Neighbors

  const ivec2 n[] = {
    ivec2(-1, -1),
    ivec2(-1, 0),
    ivec2(-1, 1),
    ivec2(0, -1),
    ivec2(0, 1),
    ivec2(1, -1),
    ivec2(1, 0),
    ivec2(1, 1)
  };

  /*
  const float w[] = {
    0.4f,
    0.4f,
    0.4f,
    2.0f,
    2.0f,
    4.0f,
    4.0f,
    4.0f
  };
  */

  const float w[] = {
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0
  };

  struct Point {
    float h;
    float d;
  } sn[8];

  int num = 0;

  for(auto &nn : n){

    ivec2 npos = ipos + nn;

    if(index.oob(npos))
      continue;

    const size_t i = index.flatten(npos);
    sn[num] = {height[i], length(vec2(nn))};
    ++num;
  }

  const size_t i = index.flatten(ipos);
  const float h = height[i];
  float transfer_tot = 0.0f;
  
  for(int i = 0; i < num; ++i){

    // Full Height-Different Between Positions!
    float diff = h - sn[i].h;
    if (diff == 0) // No Height Difference
      continue;

    // The Amount of Excess Difference!
    float excess = 0.0f;
    excess = abs(diff) - sn[i].d * w[i]* param.maxdiff;
    if (excess <= 0) // No Excess
      continue;

    excess = (diff > 0) ? -excess : excess;

    // Actual Amount Transferred
    float transfer = param.settling * excess / 2.0f;
    transfer_tot += transfer;
  }

  height_diff[i] = transfer_tot / (float) num;

}

//! Laplacian Based Mass Wasting Function
//!
__device__ void thermal_laplacian(const buffer_t<float>& height, buffer_t<float>& height_diff, const flat_t<2>& index, const ivec2 ipos, const param_t param) {

  if(index.oob(ipos))
    return;

  const size_t i = index.flatten(ipos + ivec2( 0, 0));
  float ct = height[i];
  float nx = ct;
  float ny = ct;
  float px = ct;
  float py = ct;
  if(!index.oob(ipos + ivec2( 1, 0))) px = height[index.flatten(ipos + ivec2( 1, 0))];
  if(!index.oob(ipos + ivec2(-1, 0))) nx = height[index.flatten(ipos + ivec2(-1, 0))];
  if(!index.oob(ipos + ivec2( 0, 1))) py = height[index.flatten(ipos + ivec2( 0, 1))];
  if(!index.oob(ipos + ivec2( 0,-1))) ny = height[index.flatten(ipos + ivec2( 0,-1))];

  float lx = px + nx - 2.0f * ct;
  float ly = py + ny - 2.0f * ct;
  height_diff[i] = param.settling * (lx + ly);

}

//
// Thermal Erosion Kernels
//

__global__ void compute_cascade(model_t model, buffer_t<float> transfer, const param_t param) {

  const unsigned int ind = blockIdx.x * blockDim.x + threadIdx.x;
  if(ind >= model.elem) return;
  const ivec2 ipos = model.index.unflatten(ind);

  slope_limit(model.height, transfer, model.index, ipos, param);
  //thermal_laplacian(model.height, transfer, model.index, ipos, param);

}

__global__ void apply_cascade(model_t model, buffer_t<float> transfer_b, const param_t param){
  const unsigned int ind = blockIdx.x * blockDim.x + threadIdx.x;
  if(ind >= model.elem) return;

  // Only cascade where agitation exists?

  const float transfer = transfer_b[ind];
  const float discharge = log(1.0f + model.discharge[ind])/6.0f;
  model.height[ind] += discharge * transfer;
  //model.height[ind] += transfer;
}

} // end of namespace soil

#endif