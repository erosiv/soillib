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

buffer_t<vec3> concat_impl(const buffer_t<vec2>& a, const buffer_t<float>& b);

soil::buffer concat(const buffer& a, const buffer& b){
  const buffer_t<vec2> a_t = a.as<vec2>();
  const buffer_t<float> b_t = b.as<float>();
  return soil::buffer(concat_impl(a_t, b_t));
}

//
// Index a Buffer at Integer Positions
//

soil::buffer select_index_impl(const buffer& source, const buffer_t<int>& index);

soil::buffer select_index(const soil::buffer& source, const soil::buffer& index){

  const auto index_t = index.as<int>();
  return select_index_impl(source, index_t);

}

}

#endif