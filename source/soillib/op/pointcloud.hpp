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

soil::buffer_t<vec3> sample_pointcloud_impl(const soil::buffer_t<float> &buffer, const soil::index &index, const size_t N);

soil::buffer_t<soil::vec3> sample_pointcloud(const soil::buffer_t<float>& buffer, const soil::index& index, const size_t N){

  if (buffer.elem() != index.elem())
    throw soil::error::mismatch_size(buffer.elem(), index.elem());

  if (buffer.host() != soil::host_t::GPU)
    throw soil::error::mismatch_host(buffer.host(), soil::host_t::GPU);

  return sample_pointcloud_impl(buffer, index, N);

}

}

#endif