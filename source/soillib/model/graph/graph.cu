#ifndef SOILLIB_MODEL_GRAPH_CU
#define SOILLIB_MODEL_GRAPH_CU
#define HAS_CUDA

#include <soillib/model/graph/graph.hpp>
#include <silt/op/common.hpp>
#include <math_constants.h>

namespace soil {

namespace {

inline int block(const int elem, const int thread) {
  return (elem + thread - 1) / thread;
}

}

//
// Steepest Graph Computation
//

__device__ float __length(const silt::vec2 d){
  return sqrtf(d.x*d.x + d.y*d.y);
}

template<typename DIR = D4_t>
__global__ void __steepest (
  silt::tensor_t<int> graph,          //!< Output Graph Tensor
  const silt::tensor_t<float> height, //!< Input Height Tensor
  const silt::shape shape             //!< Shape of Tensors
){

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= shape.elem)
    return;

  const silt::ivec2 ipos = shape.unflatten(n);  //!< Unflattened Position
  const DIR dir;                                //!< Direction Support
  const float hlocal = height[n];               //!< Current Lowest Height

  float smax = 0.0f;  //!< Current Steepest Slope
  int next = -1;      //!< Current Parent Node

  // Iterate over Set of Neighbors
  for(int k = 0; k < DIR::K; ++k){

    // Neighbor Position and Bounds Check
    const silt::ivec2 shift = dir.shift[k];
    const silt::ivec2 npos = ipos + shift;
    if(shape.oob(npos))
      continue;

    // Neighbor Index and Height Value
    const int nind = shape.flatten(npos);
    const float scur = (hlocal - height[nind])/__length(shift);
    if(scur > smax) {
      smax = scur;
      next = nind;
    }

  }

  // Write Steepest Node to Graph
  //  Note that if no node has a height below
  //  the local height, next points to self,
  //  as it is initialized and never updated.
  graph[n] = next;

}

//! Compute the Flow-Graph of Steepest-Neighbors for a Height-Field
silt::tensor_t<int> steepest(const silt::tensor_t<float> height, const edge_t edge) {

  if(height.host() != silt::host_t::GPU)
    throw silt::error::mismatch_host(silt::host_t::GPU, height.host());

  const auto dispatch = [&]<typename DIR>() -> silt::tensor_t<int> {
    const silt::shape shape = height.shape();
    silt::tensor_t<int> graph(shape, silt::host_t::GPU);
    __steepest<DIR><<<block(shape.elem, 512), 512>>>(graph, height, shape);
    return graph;
  };

  switch(edge){
    case D4: return dispatch.template operator()<D4_t>();
    case D8: return dispatch.template operator()<D8_t>();
    default: throw std::invalid_argument("invalid edge enumerator");
  }

}

//
// Random Direction Flow
//

__global__ void __seed(silt::tensor_t<curandState> buf, const size_t seed, const size_t offset) {
  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= buf.elem()) return;
  curand_init(seed, n, offset, &buf[n]);
}

template<typename DIR = D4_t>
__global__ void __random_weighted (
  silt::tensor_t<int> graph,          //!< Output Graph Tensor
  silt::tensor_t<curandState> rand,   //!< Random Number Generator
  const silt::tensor_t<float> height, //!< Input Height Tensor
  const silt::shape shape,            //!< Shape of Tensors
  const float T                       //!< Gibbs Distribution Temperature
){

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= shape.elem)
    return;

  const silt::ivec2 ipos = shape.unflatten(n);  //!< Unflattened Position
  const DIR dir;                                //!< Direction Support
  const float hlocal = height[n];               //!< Current Lowest Height

  // Compute the Local State Energies
  //  And the Probability Distribution
  //  Note that we assume that the state energy is the potential energy,
  //  meaning that the probability of transition is given by the height
  //  difference. Dividing by the distance makes it a rate.

  float CDF[DIR::K];        //!< Cumulative Distribution Function
  float Z = 0.0f;           //!< Distribution Partition Function

  for(int k = 0; k < DIR::K; ++k) {
  
    const silt::ivec2 shift = dir.shift[k];
    const silt::ivec2 npos = ipos + shift;
    if(shape.oob(npos))
      continue;
    const int nind = shape.flatten(npos);

    // State Transition Energy
    const float dE = (hlocal - height[nind])/__length(shift);
    const float P = (dE <= 0.0f) ? 0.0f : __expf(dE / T);
    CDF[k] = Z + P;
    Z += P;

  }

  //
  // Inverse Transform Sample the CDF
  //

  int next = -1;                                  //!< Current Parent Node
  const float uniform = curand_uniform(&rand[n]); //!< Uniform Random Variable
  for(int k = 0; k < DIR::K; ++k) {

    const silt::ivec2 shift = dir.shift[k];
    const silt::ivec2 npos = ipos + shift;
    if(shape.oob(npos))
      continue;
    const int nind = shape.flatten(npos);

    // Sample the 
    if(uniform < (CDF[k] / Z)){
      next = nind;
      break;
    }

  }

  // Write Steepest Node to Graph
  //  Note that if no node has a height below
  //  the local height, next points to self,
  //  as it is initialized and never updated.
  graph[n] = next;

}

silt::tensor_t<int> random_weighted(const silt::tensor_t<float> height, const edge_t edge, const size_t seed, const size_t offset, const float T) {

  if(height.host() != silt::host_t::GPU)
    throw silt::error::mismatch_host(silt::host_t::GPU, height.host());

  const auto dispatch = [&]<typename DIR>() -> silt::tensor_t<int> {
    const silt::shape shape = height.shape();
    silt::tensor_t<curandState> rand(shape, silt::host_t::GPU);
    silt::tensor_t<int> graph(shape, silt::host_t::GPU);
    __seed<<<block(shape.elem, 512), 512>>>(rand, seed, offset);
    __random_weighted<DIR><<<block(shape.elem, 512), 512>>>(graph, rand, height, shape, T);
    return graph;
  };

  switch(edge){
    case D4: return dispatch.template operator()<D4_t>();
    case D8: return dispatch.template operator()<D8_t>();
    default: throw std::invalid_argument("invalid edge enumerator");
  }

}

//
// Flow Direction Kernel for Debugging
//

template<typename DIR = D4_t>
__global__ void __direction (
  silt::tensor_t<int> direction,      //!< Output Graph Tensor
  const silt::tensor_t<float> height, //!< Input Height Tensor
  const silt::shape shape             //!< Shape of Tensors
){

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= shape.elem)
    return;

  const silt::ivec2 ipos = shape.unflatten(n);  //!< Unflattened Position
  const DIR dir;                                //!< Direction Support
  const float hlocal = height[n];               //!< Current Lowest Height

  float smax = 0.0f;  //!< Current Steepest Slope
  int next = -1;      //!< Current Parent Node

  // Iterate over Set of Neighbors
  for(int k = 0; k < DIR::K; ++k){

    // Neighbor Position and Bounds Check
    const silt::ivec2 shift = dir.shift[k];
    const silt::ivec2 npos = ipos + shift;
    if(shape.oob(npos))
      continue;

    const int nind = shape.flatten(npos);
    const float scur = (hlocal - height[nind])/__length(shift);
    if(scur > smax){
      smax = scur;
      next = k;
    }

  }

  // Write Steepest Node to Graph
  //  Note that if no node has a height below
  //  the local height, next points to self,
  //  as it is initialized and never updated.
  direction[n] = next;

}

//! Compute the Flow-Graph of Steepest-Neighbors for a Height-Field
silt::tensor_t<int> direction(const silt::tensor_t<float> height, const edge_t edge) {

  if(height.host() != silt::host_t::GPU)
    throw silt::error::mismatch_host(silt::host_t::GPU, height.host());

  const auto dispatch = [&]<typename DIR>() -> silt::tensor_t<int> {
    const silt::shape shape = height.shape();
    silt::tensor_t<int> flow(shape, silt::host_t::GPU);
    __direction<DIR><<<block(shape.elem, 512), 512>>>(flow, height, shape);
    return flow;
  };

  switch(edge){
    case D4: return dispatch.template operator()<D4_t>();
    case D8: return dispatch.template operator()<D8_t>();
    default: throw std::invalid_argument("invalid edge enumerator");
  }

}

//
// Slope Computation Kernel
//

__global__ void __slope (
  silt::tensor_t<float> slope,        //!< Output Graph Tensor
  const silt::tensor_t<float> tensor, //!< Input Height Tensor
  const silt::tensor_t<int> flow,     //!< Flow Tensor
  const silt::vec2 scale,             //!< Scale of Tensor Index
  const silt::shape shape             //!< Shape of Tensors
){

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= shape.elem)
    return;

  const int next = flow[n];
  if(next < 0 || next == n){
    slope[n] = 0.0f;  // unknown gradient...
    return;
  }

  const silt::vec2 ipos = shape.unflatten(n);
  const silt::vec2 npos = shape.unflatten(next);

  const float ival = tensor[n];
  const float nval = tensor[next];
  slope[n] = (nval - ival) / __length(scale*(npos - ipos));

}

silt::tensor_t<float> slope (
  const silt::tensor_t<float> tensor,
  const silt::tensor_t<int> flow,
  const silt::vec2 scale
) {

  if(tensor.host() != silt::host_t::GPU)
    throw silt::error::mismatch_host(silt::host_t::GPU, tensor.host());

  const silt::shape shape = tensor.shape();
  silt::tensor_t<float> slope(shape, silt::host_t::GPU);
  __slope<<<block(shape.elem, 512), 512>>>(slope, tensor, flow, scale, shape);
  return slope;

}

//
// Accumulation Kernel
//

//
// Compute the Donor Set and Donor Count
//

template<typename DIR = D4_t>
__global__ void __donor(
  silt::tensor_t<int> donor,        //!< Output Donor Graph
  const silt::tensor_t<int> graph,  //!< Input Receiver Graph,
  const silt::shape shape
){

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= shape.elem)
    return;

  const silt::ivec2 ipos = shape.unflatten(n);  //!< Unflattened Position
  const int next = graph[n];                    //!< Receiver Node Index
  
  const DIR dir;                    //!< Direction Support
  for(int k = 0; k < DIR::K; ++k){  //!< Iterate over set of Directions

    const silt::ivec2 shift = dir.shift[k];
    const silt::ivec2 npos = ipos + shift;
    if(shape.oob(npos))
      continue;

    const int nind = shape.flatten(npos);   //!< Neighbors Global Index
    if(nind == next)                        //!< If Neighobr is Receiver...
      donor[DIR::K*nind + k] = n;           //!< Write to Neighbor's Donorset at k
  }

}

template<typename DIR = D4_t>
__global__ void __count(
  silt::tensor_t<int> count,  //!< Output Donor Graph
  silt::tensor_t<int> donor,  //!< Output Donor Graph
  const silt::shape shape
){

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= shape.elem)
    return;

  int _count = 0;
  int _donor[DIR::K];

  for(int k = 0; k < DIR::K; ++k){
    
    const int d = donor[DIR::K*n + k];
    if(d >= 0){
      _donor[_count] = d;
      ++_count;
    }
  }

  count[n] = _count;
  for(int k = 0; k < DIR::K; ++k){
    if(k < _count)
      donor[DIR::K*n + k] = _donor[k];
    else donor[DIR::K*n + k] = -1;
  }

}

struct acc_t {
  silt::tensor_t<int> donor;    // Donor Graph
  silt::tensor_t<int> count;    // Donor Count
  silt::tensor_t<float> value;  // Local Value
};

template<typename DIR = D4_t>
__global__ void __rake_compress(
  acc_t accOut,
  const acc_t accIn, 
  const silt::shape shape
){

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= shape.elem)
    return;
  
  float value = accIn.value[n]; //!< Accumulation Value
  int count = accIn.count[n];   //!< Number of Donors
  int donors[DIR::K];           //!< Donor Indices
  
  // Developer Note: This lowers register pressure on
  //  the GPU relative to a for loop, while doing
  //  minimal work and thread stalling.
  if(count >= 1) donors[0] = accIn.donor[DIR::K*n + 0];
  if(count >= 2) donors[1] = accIn.donor[DIR::K*n + 1];
  if(count >= 3) donors[2] = accIn.donor[DIR::K*n + 2];
  if(count >= 4) donors[3] = accIn.donor[DIR::K*n + 3];
  if constexpr(std::is_same_v<DIR, D8_t>) {
    if(count >= 5) donors[4] = accIn.donor[DIR::K*n + 4];
    if(count >= 6) donors[5] = accIn.donor[DIR::K*n + 5];
    if(count >= 7) donors[6] = accIn.donor[DIR::K*n + 6];
    if(count >= 8) donors[7] = accIn.donor[DIR::K*n + 7];
  }

  // Iterate over the Set of Donors
  for(int k = 0; k < count; ++k){

    // Retrieve Donor at Index
    const int donor = donors[k];
    const int dcount = accIn.count[donor];

    // Donor is a Leaf-Node: Accumulate and Prune
    if(dcount == 0){
      value += accIn.value[donor];  // Add the Donor's Value
      donors[k] = donors[count-1];  // Move Last Donor Forward
      donors[count-1] = -1;         // Prune Last Donor
      count -= 1;                   // Shorten List
      k -= 1;                       // Repeat Iteration
    }

    // Donor has a Single Donor: Accmulate and Pointer Jump
    else if(dcount == 1){
      value += accIn.value[donor];            // Add the Donor's Value
      donors[k] = accIn.donor[DIR::K*donor];  // Pointer Jump the Donor
    }

  }

  accOut.value[n] = value;  //!< Output Value
  accOut.count[n] = count;  //!< Output Count
  if(count >= 1) accOut.donor[DIR::K*n + 0] = donors[0];
  if(count >= 2) accOut.donor[DIR::K*n + 1] = donors[1];
  if(count >= 3) accOut.donor[DIR::K*n + 2] = donors[2];
  if(count >= 4) accOut.donor[DIR::K*n + 3] = donors[3];
  if constexpr(std::is_same_v<DIR, D8_t>) {
    if(count >= 5) accOut.donor[DIR::K*n + 4] = donors[4];
    if(count >= 6) accOut.donor[DIR::K*n + 5] = donors[5];
    if(count >= 7) accOut.donor[DIR::K*n + 6] = donors[6];
    if(count >= 8) accOut.donor[DIR::K*n + 7] = donors[7];
  }
}

//! Compute the Upstream Accumulation of a Field
//!\todo Possible Optimization: Per-Cell Ping-Pong Scheme for fewer writes
silt::tensor_t<float> accumulate (
  const silt::tensor_t<int> graph,
  const silt::tensor_t<float> value,
  const edge_t edge
){

  const auto dispatch = [&]<typename DIR>() -> silt::tensor_t<float> {

    const silt::shape shape = graph.shape();
    const silt::shape dshape = silt::shape(shape[0], shape[1], DIR::K);

    acc_t accA, accB;
    accA.count = silt::tensor_t<int>(shape, silt::host_t::GPU);
    accA.value = silt::tensor_t<float>(shape, silt::host_t::GPU);
    
    accB.count = silt::tensor_t<int>(shape, silt::host_t::GPU);
    accB.value = silt::tensor_t<float>(shape, silt::host_t::GPU);

    accA.donor = silt::tensor_t<int>(dshape, silt::host_t::GPU);
    accB.donor = silt::tensor_t<int>(dshape, silt::host_t::GPU);

    silt::set(accA.donor, -1);
    silt::set(accA.value, value);
    __donor<DIR><<<block(shape.elem, 512), 512>>>(accA.donor, graph, shape);
    __count<DIR><<<block(shape.elem, 512), 512>>>(accA.count, accA.donor, shape);

    // Execute Rake-Compression Iterations
    const size_t iter = std::ceil(std::log2f((float)shape.elem)/2.0f);
    for(size_t i = 0; i <= iter; ++i){
      __rake_compress<DIR><<<block(shape.elem, 256), 256>>>(accB, accA, shape);
      __rake_compress<DIR><<<block(shape.elem, 256), 256>>>(accA, accB, shape);
    }
    cudaDeviceSynchronize();

    return accA.value;

  };
  
  switch(edge){
    case D4: return dispatch.template operator()<D4_t>();
    case D8: return dispatch.template operator()<D8_t>();
    default: throw std::invalid_argument("invalid edge enumerator");
  }
  
}

} // end of namespace soil

#endif