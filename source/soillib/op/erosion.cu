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
// Erosion Function
//

void erode(map_t& map, data_t& data, const param_t param, const size_t steps) {

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
  
  // Allocate Estimate Buffers for Transported Quantities
  data.mass_track       = soil::buffer_t<float>(data.mass.elem(), soil::host_t::GPU);
  data.discharge_track  = soil::buffer_t<float>(data.discharge.elem(), soil::host_t::GPU);
  data.momentum_track   = soil::buffer_t<vec2>(data.momentum.elem(), soil::host_t::GPU);

  data.debris_track           = soil::buffer_t<float>(data.debris.elem(), soil::host_t::GPU);
  data.debris_momentum_track  = soil::buffer_t<vec2>(data.debris_momentum.elem(), soil::host_t::GPU);

  //
  // Execute Solution
  //

  for(size_t step = 0; step < steps; ++step){

    // Reset Estimates
    set(data.discharge_track, 0.0f);
    set(data.momentum_track, vec2(0.0f));
    set(data.mass_track, 0.0f);
    set(data.debris_track, 0.0f);
    set(data.debris_momentum_track, vec2(0.0f));
    cudaDeviceSynchronize();

    // Solve Estimates
    fluvial::solve<<<block(n_samples, 512), 512>>>(map, data, n_samples, param);
    debris::solve<<<block(n_samples, 512), 512>>>(map, data, n_samples, param);
    cudaDeviceSynchronize();

    //
    // Debris Flow Kernel
    //

    // Filter Estimates
    filter(data.momentum, data.momentum_track, param.lrate);
    filter(data.discharge, data.discharge_track, param.lrate);
    filter(data.mass, data.mass_track, param.lrate);
    filter(data.debris, data.debris_track, param.lrate);
    filter(data.debris_momentum, data.debris_momentum_track, param.lrate);
    cudaDeviceSynchronize();

    // Execute Height-Map Mass-Transfer
    fluvial::mt<<<block(map.elem, 512), 512>>>(map, data, param);
    debris::mt<<<block(map.elem, 512), 512>>>(map, data, param);

    // Increment Model Age for Rand-State Initialization
    map.age++;

  }

}

} // end of namespace soil

#endif