#ifndef SOILLIB_OP_POINTCLOUD
#define SOILLIB_OP_POINTCLOUD

// Pointcloud Operations
//
//  We are introducing this operation set, so that we can
//  detach ourselves from requiring a grid for property storage.
//  The lookup and interpolation procedure for the pointset can
//  be thought of as an index type. For now, we will NOT implement
//  it through the index interface since it would require abstraction.
//  
//  1. Generate a Random Sample of Positions in an Index
//  2. Sample Values at Positions into Buffer
//    -> Note: These operations could be merged into a single pointcloud sample
//    -> This does NOT work for more complex data types stored in buffer.
//    -> Note: This should of course interpolate the lookup correctly.
//    -> Optionally, we can also sample the normal value (i.e. the gradient).
//  3. Write a Radial Basis Function Interpolation Type
//  4. Re-Sample the RBF Interpolation into a Grid!
//  5. Implement the RBF Lookup Gradient as well, for Surface Normal
// 
//  That would be the first set of steps.
//  The structure can then be traced, as well as descended on and we can implement
//  area accumulation on the grid and later potentially erosion. We start with the
//  underlying data-structures first.

#include <soillib/core/buffer.hpp>
#include <soillib/core/index.hpp>

namespace soil {

//
// Sample Random Positions within the Domain
//

buffer_t<vec2> sample_N_impl(const flat_t<2>& index, const size_t N);

//! Sample N Random Positions within an Index Space
soil::buffer sample_N(const soil::index &index, const size_t N){
  const auto index_t = index.as<flat_t<2>>();
  return soil::buffer(sample_N_impl(index_t, N));
}

//
// Linear Interpolation Index
//

//! Lerp the Field at Positions
buffer_t<float> sample_lerp_impl(const buffer_t<float>& field, const flat_t<2>& index, const buffer_t<vec2>& pos);

soil::buffer sample_lerp(const buffer& field, const soil::index &index, const buffer& pos){
  const auto index_t = index.as<flat_t<2>>();
  const auto field_t = field.as<float>();
  const auto pos_t = pos.as<vec2>();
  return soil::buffer(sample_lerp_impl(field_t, index_t, pos_t));
}

//
// Concatenate two Buffers (Copy)
//



//
// Index a Buffer at Integer Positions
//





//
// Pointcloud Direct Methods
//

soil::buffer_t<vec3> pointcloud_sample_impl(const soil::buffer_t<float> &buffer, const soil::index &index, const size_t N);

soil::buffer pointcloud_sample(const soil::buffer& buffer, const soil::index& index, const size_t N){

  if (buffer.elem() != index.elem())
  throw soil::error::mismatch_size(buffer.elem(), index.elem());
  
  if (buffer.host() != soil::host_t::GPU)
  throw soil::error::mismatch_host(buffer.host(), soil::host_t::GPU);

  const auto buffer_t = buffer.as<float>();
  return soil::buffer(pointcloud_sample_impl(buffer_t, index, N));

}

}

#endif