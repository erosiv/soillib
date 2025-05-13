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

void erode(model_t& model, const param_t param, const size_t steps) {

  //
  // Initialize Rand-State Buffer (One Per Sample)
  //
  // note: the offset in the sequence should be number of times rand is sampled
  // that way the sampling procedure becomes deterministic

  const size_t n_samples = param.samples;

  if(model.rand.elem() != n_samples){
    model.rand = soil::buffer_t<curandState>(n_samples, soil::host_t::GPU);
    seed(model.rand, 0, 2 * model.age);
  }
  
  // Allocate Estimate Buffers for Transported Quantities
  model.mass_track = soil::buffer_t<float>(model.mass.elem(), soil::host_t::GPU);
  model.discharge_track = soil::buffer_t<float>(model.discharge.elem(), soil::host_t::GPU);
  model.momentum_track = soil::buffer_t<vec2>(model.momentum.elem(), soil::host_t::GPU);

  model.debris_track = soil::buffer_t<float>(model.debris.elem(), soil::host_t::GPU);
  model.debris_momentum_track = soil::buffer_t<vec2>(model.debris_momentum.elem(), soil::host_t::GPU);

  //
  // Execute Solution
  //

  for(size_t step = 0; step < steps; ++step){

    // Reset Estimates
    set(model.discharge_track, 0.0f);
    set(model.momentum_track, vec2(0.0f));
    set(model.mass_track, 0.0f);
    // set(model.debris_track, 0.0f);
    // set(model.debris_momentum_track, vec2(0.0f));
    cudaDeviceSynchronize();

    // Solve Estimates
    fluvial::solve<<<block(n_samples, 512), 512>>>(model, n_samples, param);
    // debris::solve<<<block(n_samples, 512), 512>>>(model, n_samples, param);
    cudaDeviceSynchronize();

    //
    // Debris Flow Kernel
    //

    // Filter Estimates
    filter(model.momentum, model.momentum_track, param.lrate);
    filter(model.discharge, model.discharge_track, param.lrate);
    filter(model.mass, model.mass_track, param.lrate);
    // filter(model.debris, model.debris_track, param.lrate);
    // filter(model.debris_momentum, model.debris_momentum_track, param.lrate);
    cudaDeviceSynchronize();

    // Execute Height-Map Mass-Transfer
    fluvial::mt<<<block(model.height.elem(), 512), 512>>>(model, param);
    // debris::mt<<<block(model.height.elem(), 512), 512>>>(model, param);

    // Increment Model Age for Rand-State Initialization
    model.age++;

  }

}

} // end of namespace soil

#endif