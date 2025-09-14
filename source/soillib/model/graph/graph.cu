#ifndef SOILLIB_MODEL_GRAPH_CU
#define SOILLIB_MODEL_GRAPH_CU
#define HAS_CUDA

#include <soillib/model/graph/graph.hpp>
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
    silt::ivec2(-1,  0),
    silt::ivec2( 0, -1),
    silt::ivec2( 0,  1),
    silt::ivec2( 1,  0)
  }{}
  const silt::ivec2 shift[K];
};

//! D8 Connectivity Static Struct
struct D8 {
  static constexpr const size_t K = 8;
  __device__ D8():shift{
    silt::ivec2(-1, -1),
    silt::ivec2(-1,  0),
    silt::ivec2(-1,  1),
    silt::ivec2( 0, -1),
    silt::ivec2( 0,  1),
    silt::ivec2( 1, -1),
    silt::ivec2( 1,  0),
    silt::ivec2( 1,  1)
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
    const int hcur = height[nind];
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

} // end of namespace soil

#endif