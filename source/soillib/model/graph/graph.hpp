#ifndef SOILLIB_MODEL_GRAPH
#define SOILLIB_MODEL_GRAPH

#include <silt/core/types.hpp>
#include <silt/core/shape.hpp>
#include <silt/core/tensor.hpp>

namespace soil {

//! Enumerator for Edge Connectivity Types
enum edge_t {
  D4 = 0, //!< D4 Graph Edge Connectivity
  D8 = 1  //!< D8 Graph Edge Connectivity
};

//
// Neighbor Connectivity Structs
//

//! D4 Connectivity Static Struct
struct D4_t {
  static constexpr const size_t K = 4;
  __device__ D4_t():shift{
    silt::ivec2(-1.0,  0.0),
    silt::ivec2( 0.0, -1.0),
    silt::ivec2( 0.0,  1.0),
    silt::ivec2( 1.0,  0.0)
  }{}
  const silt::ivec2 shift[K];
};

//! D8 Connectivity Static Struct
struct D8_t {
  static constexpr const size_t K = 8;
  __device__ D8_t():shift{
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

//! Compute the Direction of the FLow
silt::tensor_t<int> direction(const silt::tensor_t<float> height, const edge_t edge);

//! Compute the Flow-Graph of Steepest-Neighbors for a Height-Field
silt::tensor_t<int> steepest(const silt::tensor_t<float> height, const edge_t edge);

silt::tensor_t<int> random_weighted(const silt::tensor_t<float> height, const edge_t edge, const size_t seed, const size_t offset, const float T);

//! Compute the Upstream Accumulation of a Field
silt::tensor_t<float> accumulate(const silt::tensor_t<int> graph, const silt::tensor_t<float> field, const edge_t edge);

} // end of namespace soil

#endif