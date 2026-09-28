#pragma once

#include <soillib/soillib.hpp>

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
    silt::ivec2(-1.0,  0.0),
    silt::ivec2( 0.0, -1.0),
    silt::ivec2( 0.0,  1.0),
    silt::ivec2( 1.0,  0.0),
    silt::ivec2(-1.0, -1.0),
    silt::ivec2(-1.0,  1.0),
    silt::ivec2( 1.0, -1.0),
    silt::ivec2( 1.0,  1.0)
  }{}
  const silt::ivec2 shift[K];
};

//! Compute the Flow-Graph of Steepest-Neighbors for a Height-Field
silt::tensor_t<int> steepest(const silt::tensor_t<float> height, const edge_t edge);

//! Sample a Slope-Weighted Descending Receiver Graph.
//! Each cell draws one receiver from the discrete distribution over its
//! descending neighbors, with probability proportional to the downhill
//! slope. Every draw is a single-receiver graph the O(N) accumulation
//! consumes unchanged, but the expected receiver distribution is the
//! slope-weighted MFD partition -- so averaging draws recovers a
//! multi-flow field without ever building one, breaking up the
//! directional artifacts a deterministic steepest-descent graph locks in.
//! `rng` is advanced in place: seed it once (silt::seed / silt.rand) and
//! reuse it across draws for a controlled, reproducible sequence.
silt::tensor_t<int> random_weighted(const silt::tensor_t<float> height, const edge_t edge, silt::tensor_t<silt::rng> rng);

//! Compute the Upstream Accumulation of a Source Field
silt::tensor_t<float> accumulate(const silt::tensor_t<int> graph, const silt::tensor_t<float> source, const edge_t edge);

//! Compute the Upstream Accumulation of a Source Field (w. Decay)
silt::tensor_t<float> accumulate_decay(const silt::tensor_t<int> graph, const silt::tensor_t<float> source, const silt::tensor_t<float> decay, const edge_t edge);

} // end of namespace soil
