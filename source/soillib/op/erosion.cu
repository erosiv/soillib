#ifndef SOILLIB_NODE_EROSION_CU
#define SOILLIB_NODE_EROSION_CU
#define HAS_CUDA

#include <soillib/util/error.hpp>

#include <cuda_runtime.h>
#include <math_constants.h>
#include <iostream>

#include <soillib/op/gather.hpp>
#include <soillib/op/erosion.hpp>

#include <soillib/op/cu_common.cu>

#include <soillib/op/erosion_map.cu>
#include <soillib/op/erosion_fluvial.cu>
#include <soillib/op/erosion_thermal.cu>

namespace soil {

//
// Uplift Application
//

__global__ void uplift(map_t map, const param_t param) {

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= map.elem)
    return;

  const vec3 scale = map.scale * 1E3f;    // Cell Scale [m] (conv. from km)
  const float dt = param.timeStep;        //!< Geological Timestep [y] 
  const float uplift = param.uplift;      //!< Uplift Rate [m/y]
  const float mask = map.uplift[n];       //!< Uplift Mask

  map.height[n] += dt * mask * uplift / scale.z; //!< Total Height Delta

}

//
// Erosion Function
//

void erode(map_grid& map, data_t& data, data_t& track, const param_t param, const size_t steps) {

  //
  // Initialize Rand-State Buffer (One Per Sample)
  //
  // note: the offset in the sequence should be number of times rand is sampled
  // that way the sampling procedure becomes deterministic

  const size_t n_samples = param.samples;
  if(map.rand.elem() != n_samples){
    map.rand = soil::buffer_t<curandState>(n_samples, soil::host_t::GPU);
    seed(map.rand, 0, 4 * map.age);
  }

  //
  // Execute Solution
  //

  for(size_t step = 0; step < steps; ++step){

    // Reset Estimates
    set(track.discharge, 0.0f);
    set(track.momentum, vec2(0.0f));
    set(track.mass, 0.0f);
    set(track.debris, 0.0f);
    set(track.debris_momentum, vec2(0.0f));
    cudaDeviceSynchronize();

    // Solve Estimates
    fluvial::solve<<<block(n_samples, 512), 512>>>(map, data, track, n_samples, param);
    debris::solve<<<block(n_samples, 512), 512>>>(map, data, track, n_samples, param);
    cudaDeviceSynchronize();

    // Filter Estimates
    filter(data.momentum, track.momentum, param.lrate);
    filter(data.discharge, track.discharge, param.lrate);
    filter(data.mass, track.mass, param.lrate);
    filter(data.debris, track.debris, param.lrate);
    filter(data.debris_momentum, track.debris_momentum, param.lrate);
    cudaDeviceSynchronize();

    // Execute Height-Map Mass-Transfer
    fluvial::mt<<<block(map.elem, 512), 512>>>(map, data, param);
    debris::mt<<<block(map.elem, 512), 512>>>(map, data, param);
    uplift<<<block(map.elem, 512), 512>>>(map, param);

    // Increment Model Age for Rand-State Initialization
    map.age++;

  }

}

} // end of namespace soil

#endif