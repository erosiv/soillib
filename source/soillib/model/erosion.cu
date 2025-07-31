#ifndef SOILLIB_MODEL_EROSION_CU
#define SOILLIB_MODEL_EROSION_CU
#define HAS_CUDA

#include <soillib/soillib.hpp>
#include <soillib/core/error.hpp>
#include <soillib/core/tensor.hpp>

#include <soillib/op/gather.hpp>
#include <soillib/op/common.hpp>

#include <soillib/model/erosion.hpp>
#include <soillib/model/erosion_map.cu>
#include <soillib/model/erosion_fluvial.cu>
// #include <soillib/model/erosion_thermal.cu>

#include <math_constants.h>

namespace soil {

//
// Uplift Application
//  Note: Also applies erosion to map

__global__ void uplift(map_t map, const param_t param) {

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= map.elem)
    return;

  const vec3 scale = map.scale * 1E3f;    // Cell Scale [m] (conv. from km)
  const vec2 cl = vec2(scale.x, scale.y); // Cell Length [m, m]
  const float Ac = scale.x*scale.y;       // Cell Area [m^2]
  const float Z = Ac * scale.z;           // Height Conversion [m^3]

  const float transfer = map.transfer[n];
  const vec2 pos = __topos(map, n);
  __transfer(map, pos, transfer, Z);

  const float dt = param.timeStep;        //!< Geological Timestep [y] 
  const float uplift = param.uplift;      //!< Uplift Rate [m/y]
  const float mask = map.uplift[n];       //!< Uplift Mask

  map.height[n] += dt * mask * uplift / scale.z; //!< Total Height Delta

}

//
// Erosion Function
//

void erode(map_t& map, data_t& data, data_t& track, const param_t param, const size_t steps) {

  //
  // Initialize Rand-State Buffer (One Per Sample)
  //
  // note: the offset in the sequence should be number of times rand is sampled
  // that way the sampling procedure becomes deterministic

  const size_t n_samples = param.samples;
  if(map.rand.elem() != n_samples){
    map.rand = soil::tensor_t<curandState>(soil::shape(n_samples), soil::host_t::GPU);
    soil::seed(map.rand, 0, 4 * map.age);
  }

  if(map.transfer.elem() != map.elem){
    map.transfer = soil::tensor_t<float>(map.shape, soil::host_t::GPU);
  }

  const scale_t scale(map.scale);

  //
  // Execute Solution
  //

  for(size_t step = 0; step < steps; ++step){

    // Reset Estimates
    soil::set(track.discharge, 0.0f);
    soil::set(track.momentum, 0.0f);
    soil::set(track.mass, 0.0f);
    soil::set(track.debris, 0.0f);
    soil::set(track.debris_momentum, 0.0f);
    cudaDeviceSynchronize();

//    // Solve Estimates
    fluvial::solve<<<block(n_samples, 512), 512>>>(map, data, track, n_samples, param, scale);
//    debris::solve<<<block(n_samples, 512), 512>>>(map, data, track, n_samples, param);
    cudaDeviceSynchronize();

    // Filter Estimates
    soil::mix(data.momentum, track.momentum, param.lrate);
    soil::mix(data.discharge, track.discharge, param.lrate);
    soil::mix(data.mass, track.mass, param.lrate);
    soil::mix(data.debris, track.debris, param.lrate);
    soil::mix(data.debris_momentum, track.debris_momentum, param.lrate);
    cudaDeviceSynchronize();

    // Execute Height-Map Mass-Transfer
    soil::set(map.transfer, 0.0f);
    fluvial::mt<<<block(map.elem, 512), 512>>>(map, data, param, scale);
//    debris::mt<<<block(map.elem, 512), 512>>>(map, data, param);
    uplift<<<block(map.elem, 512), 512>>>(map, param);

    // Increment Model Age for Rand-State Initialization
    map.age++;

  }

}

} // end of namespace soil

#endif