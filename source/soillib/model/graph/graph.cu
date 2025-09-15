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
// Neighbor Connectivity Structs
//

//! D4 Connectivity Static Struct
struct D4 {
  static constexpr const size_t K = 4;
  __device__ D4():shift{
    silt::ivec2(-1.0,  0.0),
    silt::ivec2( 0.0, -1.0),
    silt::ivec2( 0.0,  1.0),
    silt::ivec2( 1.0,  0.0)
  }{}
  const silt::ivec2 shift[K];
};

//! D8 Connectivity Static Struct
struct D8 {
  static constexpr const size_t K = 8;
  __device__ D8():shift{
    silt::ivec2(-1.0, -1.0),
    silt::ivec2(-1.0,  0.0),
    silt::ivec2(-1.0,  1.0),
    silt::ivec2( 0.0, -1.0),
    silt::ivec2( 0.0,  1.0),
    silt::ivec2( 1.0, -1.0),
    silt::ivec2( 1.0,  0.0),
    silt::ivec2( 1.0,  1.0)
  }{}
  const silt::ivec2 shift[K];
};

//
// Steepest Graph Computation
//

template<typename DIR = D4>
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

  float hmin = height[n]; //!< Current Lowest Height
  int next = n;           //!< Current Parent Node

  // Iterate over Set of Neighbors
  for(int k = 0; k < DIR::K; ++k){

    // Neighbor Position and Bounds Check
    const silt::ivec2 shift = dir.shift[k];
    const silt::ivec2 npos = ipos + shift;
    if(shape.oob(npos))
      continue;

    // Neighbor Index and Height Value
    const int nind = shape.flatten(npos);
    const float hcur = height[nind];
    if(hcur < hmin){
      hmin = hcur;
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
silt::tensor_t<int> steepest(const silt::tensor_t<float> height) {

  if(height.host() != silt::host_t::GPU)
    throw silt::error::mismatch_host(silt::host_t::GPU, height.host());

  const silt::shape shape = height.shape();
  silt::tensor_t<int> graph(shape, silt::host_t::GPU);
  __steepest<<<block(shape.elem, 512), 512>>>(graph, height, shape);
  return graph;

}

//
// Flow Direction Kernel for Debugging
//

template<typename DIR = D4>
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

  float hmin = height[n]; //!< Current Lowest Height
  int next = -1;           //!< Current Parent Node

  // Iterate over Set of Neighbors
  for(int k = 0; k < DIR::K; ++k){

    // Neighbor Position and Bounds Check
    const silt::ivec2 shift = dir.shift[k];
    const silt::ivec2 npos = ipos + shift;
    if(shape.oob(npos))
      continue;

    const int nind = shape.flatten(npos);
    const float hcur = height[nind];
    if(hcur < hmin){
      hmin = hcur;
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
silt::tensor_t<int> direction(const silt::tensor_t<float> height) {

  if(height.host() != silt::host_t::GPU)
    throw silt::error::mismatch_host(silt::host_t::GPU, height.host());

  const silt::shape shape = height.shape();
  silt::tensor_t<int> flow(shape, silt::host_t::GPU);
  __direction<<<block(shape.elem, 512), 512>>>(flow, height, shape);
  return flow;

}

//
// Accumulation Kernel
//

//
// Compute the Donor Set and Donor Count
//

template<typename DIR = D4>
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

template<typename DIR = D4>
__global__ void __count(
  silt::tensor_t<int> count,  //!< Output Donor Graph
  silt::tensor_t<int> donor,  //!< Output Donor Graph
  const silt::shape shape
){

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= shape.elem)
    return;

  const DIR dir;                                //!< Direction Support
  int _count = 0;
  int _donor[DIR::K]{-1, -1, -1, -1};

  for(int k = 0; k < DIR::K; ++k){
    
    const int d = donor[DIR::K*n + k];
    if(d >= 0){
      _donor[_count] = d;
      ++_count;
    }
  }

  count[n] = _count;
  for(int k = 0; k < DIR::K; ++k){
    donor[DIR::K*n + k] = _donor[k];
  }

}

struct acc_t {
  silt::tensor_t<int> donor;    // Donor Graph
  silt::tensor_t<int> count;    // Donor Count
  silt::tensor_t<float> value;  // Local Value
};

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
  int donors[4];                //!< Donor Indices
  
  // Developer Note: This lowers register pressure on
  //  the GPU relative to a for loop, while doing
  //  minimal work and thread stalling.
  if(count >= 1) donors[0] = accIn.donor[4*n + 0];
  if(count >= 2) donors[1] = accIn.donor[4*n + 1];
  if(count >= 3) donors[2] = accIn.donor[4*n + 2];
  if(count >= 4) donors[3] = accIn.donor[4*n + 3];

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
      value += accIn.value[donor];      // Add the Donor's Value
      donors[k] = accIn.donor[4*donor]; // Pointer Jump the Donor
    }

  }

  accOut.value[n] = value;  //!< Output Value
  accOut.count[n] = count;  //!< Output Count
  if(count >= 1) accOut.donor[4*n + 0] = donors[0];
  if(count >= 2) accOut.donor[4*n + 1] = donors[1];
  if(count >= 3) accOut.donor[4*n + 2] = donors[2];
  if(count >= 4) accOut.donor[4*n + 3] = donors[3];

}

//! Compute the Upstream Accumulation of a Field
//!\todo Possible Optimization: Per-Cell Ping-Pong Scheme for fewer writes
silt::tensor_t<float> accumulate(const silt::tensor_t<int> graph, const silt::tensor_t<float> value){

  const silt::shape shape = graph.shape();
  const silt::shape dshape = silt::shape(shape[0], shape[1], D4::K);

  // Construct the Donor Set and Count
  acc_t accA, accB;

  accA.donor = silt::tensor_t<int>(dshape, silt::host_t::GPU);
  accA.count = silt::tensor_t<int>(shape, silt::host_t::GPU);
  accA.value = silt::tensor_t<float>(shape, silt::host_t::GPU);
  
  accB.donor = silt::tensor_t<int>(dshape, silt::host_t::GPU);
  accB.count = silt::tensor_t<int>(shape, silt::host_t::GPU);
  accB.value = silt::tensor_t<float>(shape, silt::host_t::GPU);

  silt::set(accA.donor, -1);
  silt::set(accA.value, value);
  __donor<<<block(shape.elem, 512), 512>>>(accA.donor, graph, shape);
  __count<<<block(shape.elem, 512), 512>>>(accA.count, accA.donor, shape);

  // Execute Rake-Compression Iterations
  const size_t iter = std::ceil(std::log2f((float)shape.elem)/2.0f);
  for(size_t i = 0; i < iter; ++i){
    __rake_compress<<<block(shape.elem, 256), 256>>>(accB, accA, shape);
    __rake_compress<<<block(shape.elem, 256), 256>>>(accA, accB, shape);
  }
  cudaDeviceSynchronize();

  return accA.value;

}

} // end of namespace soil

#endif