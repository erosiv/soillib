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

soil::buffer_t<vec3> pointcloud_sample_impl(const soil::buffer_t<float> &buffer, const soil::index &index, const size_t N);

soil::buffer pointcloud_sample(const soil::buffer& buffer, const soil::index& index, const size_t N){

  if (buffer.elem() != index.elem())
  throw soil::error::mismatch_size(buffer.elem(), index.elem());
  
  if (buffer.host() != soil::host_t::GPU)
  throw soil::error::mismatch_host(buffer.host(), soil::host_t::GPU);

  const auto buffer_t = buffer.as<float>();
  return soil::buffer(pointcloud_sample_impl(buffer_t, index, N));

}

soil::buffer_t<vec3> pointcloud_normal_impl(const soil::buffer_t<float> &height, const soil::buffer_t<vec3> &pos, const soil::index &index, const vec3 scale);

soil::buffer pointcloud_normal(const soil::buffer& height, const soil::buffer& pos, const soil::index& index, const vec3 scale){

  if (height.host() != soil::host_t::GPU)
  throw soil::error::mismatch_host(height.host(), soil::host_t::GPU);

  if (pos.host() != soil::host_t::GPU)
  throw soil::error::mismatch_host(pos.host(), soil::host_t::GPU);

  const auto height_t = height.as<float>();
  const auto pos_t = pos.as<vec3>();
  return soil::buffer(pointcloud_normal_impl(height_t, pos_t, index, scale));

}

void pointcloud_scale_impl(const soil::buffer_t<vec3> &buffer, const soil::index &index, const soil::vec3 scale);

void pointcloud_scale(const soil::buffer& buffer, const soil::index& index, const soil::vec3 scale){

  if (buffer.host() != soil::host_t::GPU)
  throw soil::error::mismatch_host(buffer.host(), soil::host_t::GPU);

  const auto buffer_t = buffer.as<vec3>();
  pointcloud_scale_impl(buffer_t, index, scale);

}

//
// KDTree Wrapper
//

soil::buffer_t<vec3> knnquery(const soil::buffer_t<vec3>& data, const soil::buffer_t<vec3>& query, const size_t k);
void knnbuild(soil::buffer_t<vec3>& data);

struct kdtree {
  
  kdtree(soil::buffer& buffer):
    elem(buffer.elem()){

    if(buffer.type() != soil::VEC3)
      throw soil::error::mismatch_type(buffer.type(), soil::VEC3);
    
    this->buffer = buffer.as<vec3>();
    knnbuild(this->buffer);

  }

  buffer knn(const buffer& query, const size_t k) const {

//    if(query.type() != soil::VEC3)
//      throw soil::error::mismatch_type(query.type(), soil::VEC3);

    const auto query_t = query.as<vec3>();
    return soil::buffer(knnquery(this->buffer, query_t, k));

  }

  const size_t elem;

private:
  soil::buffer_t<vec3> buffer;
};

}

#endif