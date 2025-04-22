#ifndef SOILLIB_OP_POINTCLOUD
#define SOILLIB_OP_POINTCLOUD
#define HAS_CUDA

#include <soillib/soillib.hpp>
#include <soillib/core/buffer.hpp>
#include <soillib/core/index.hpp>

#include <curand_kernel.h>
#include <soillib/op/common.hpp>
#include <soillib/op/gather.hpp>

#include <soillib/op/pointcloud.hpp>
#include <soillib/index/kdtree.hpp>

#include <cukd/builder.h>
#include <cukd/knn.h>

namespace soil {

namespace {

__global__ void seed(buffer_t<curandState> buffer, const size_t seed, const size_t offset) {

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= buffer.elem()) return;

  curand_init(seed, n, offset, &buffer[n]);

}

}

//
// Random Position Sampling within Index Bound
//

__global__ void _sample_N(soil::buffer_t<vec2> output, soil::buffer_t<curandState> rand, const soil::flat_t<2> index, const size_t N){

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= N) return;

  curandState* randState = &rand[n];
  output[n] = vec2 {
    curand_uniform(randState)*float(index[0]-1),
    curand_uniform(randState)*float(index[1]-1)
  };

}

soil::buffer_t<vec2> sample_N_impl(const soil::flat_t<2> &index, const size_t N){

  soil::buffer_t<curandState> rand(N, soil::host_t::GPU);
  seed<<<block(N, 1024), 1024>>>(rand, 0, 0);
  cudaDeviceSynchronize();

  soil::buffer_t<vec2> output(N, soil::GPU);
  _sample_N<<<block(N, 1024), 1024>>>(output, rand, index, N);

  return output;

}

//
// Sample Lerp Implementation
//

__global__ void _sample_lerp(const soil::buffer_t<float> field, soil::buffer_t<float> output, const soil::flat_t<2> index, const soil::buffer_t<vec2> pos_b){

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= pos_b.elem()) return;

  vec2 pos = pos_b[n];
  
  lerp_t<float> lerp = gather(field, index, pos);
  output[n] = lerp.val();

}

soil::buffer_t<float> sample_lerp_impl(const soil::buffer_t<float> &field, const soil::flat_t<2> &index, const soil::buffer_t<vec2>& pos){

  const size_t elem = pos.elem();
  soil::buffer_t<float> output(elem, soil::GPU);
  _sample_lerp<<<block(elem, 1024), 1024>>>(field, output, index, pos);

  return output;

}

//
// Buffer Concatenation
//

__global__ void _concat(soil::buffer_t<vec3> output, const soil::buffer_t<vec2> a_t, const soil::buffer_t<float> b_t){

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= output.elem()) return;

  vec2 a = a_t[n];
  float b = b_t[n];
  output[n] = vec3(a.x, a.y, b);

}

buffer_t<vec3> concat_impl(const buffer_t<vec2>& a, const buffer_t<float>& b){
  const size_t elem = a.elem();
  soil::buffer_t<vec3> output(elem, soil::GPU);
  _concat<<<block(elem, 1024), 1024>>>(output, a, b);
  return output;
}

//
// Index Based Buffer Select
//

template<typename T>
__global__ void _select_index(soil::buffer_t<T> output, const soil::buffer_t<T> source, const soil::buffer_t<int> index_b){

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= output.elem()) return;

  const int index = index_b[n];
  output[n] = source[index];

}

buffer select_index_impl(const buffer& buffer, const buffer_t<int>& index){
  return soil::select(buffer.type(), [&]<typename T>() -> soil::buffer {

    const auto buffer_t = buffer.as<T>();
    const size_t elem = index.elem();
    soil::buffer_t<T> output(elem, soil::GPU);

    std::cout<<"impl called converted"<<std::endl;

    _select_index<<<block(elem, 1024), 1024>>>(output, buffer_t, index);
    return soil::buffer(output);

  });
}

//
// Sparse Accumulation
//

//! Accumulation Initialization:
//! In principle the assigned value should be the
//! local region of influence of each point, but
//! we don't know that value, so we set it to zero.
//! A different initialization method might have a value.
__global__ void _init_acc(soil::buffer_t<float> acc){

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= acc.elem()) return;
  acc[n] = 0.0f;

}

struct payload_traits: 
  public cukd::default_data_traits<kdtree::pnt_t> {
 
  using point_t = kdtree::pnt_t;

  static inline __device__ __host__
  point_t get_point(const payload_t &data)
  { return data.p; }

  static inline __device__ __host__
  float get_coord(const payload_t &data, int dim)
  { return cukd::get_coord(get_point(data),dim); }

  enum { has_explicit_dim = false };
  static inline __device__ int  get_dim(const payload_t &) { return -1; }

};

template<size_t K>
__device__ void knn(const soil::kdtree& kdtree, const vec2 pos, cukd::FixedCandidateList<K>& list) {

  cukd::stackBased::knn<
    cukd::FixedCandidateList<K>,
    payload_t,
    payload_traits
  >(list, make_point(pos), kdtree.data.data(), kdtree.elem());

}

/*

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= N) return;

  // Initialize Position in Domain
  curandState* randState = &rand[n];

  int cur = curand_uniform(randState)*points.elem();
  vec3 pos = points[cur];
  atomicAdd(&acc[cur], 1.0f);

  const size_t K = 4;
  cukd::FixedCandidateList<K> list(100.0);

  int maxstep = 8192;
  while(maxstep > 0){
    maxstep--;
    
    knn<K>(kdtree, vec2(pos.x, pos.y), list);
    
    int next = -1;
    for(int i = 0; i < K; ++i){
      int ind = list.get_pointID(i);
      if(ind >= 0){

        const float n = kdtree.data[ind].i;
        const vec3 np = points[n];
        if(np.z < pos.z){
          pos = np;
          next = n;
        }

      }

    }
  
    if(next == -1){
      return;
    }
  
    atomicAdd(&acc[next], 1.0f);

  }
*/

//! Greedy Descent
__global__ void sparse_descend(const soil::kdtree kdtree, const soil::buffer_t<vec3> points, soil::buffer_t<float> acc, soil::buffer_t<curandState> rand, const size_t N, const soil::flat_t<2> index) {

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= N) return;

  // Initialize Position in Domain
  curandState* randState = &rand[n];
  int ind = curand_uniform(randState)*(points.elem()-1);
  vec3 pos;
  
  const size_t K = 16;
  cukd::FixedCandidateList<K> list(100.0);
  
  int maxstep = 8192;
  while(maxstep > 0){
    maxstep--;

    pos = points[ind];
    atomicAdd(&acc[ind], 1.0f);

    knn<K>(kdtree, vec2(pos.x, pos.y), list);
    
    // Closest 3 Point Indices
    int next = -1;
    float nh = pos.z; // lowest height

    for(int i = 0; i < K; ++i){

      int nk = list.get_pointID(i);
      if(nk >= 0){

        if(nk == ind)
          continue;

        const float n = kdtree.data[nk].i;
        const vec3 np = points[n];

        if(np.z < nh){
          nh = np.z;
          next = n;
        }
      }
    }
  
    if(next == -1){
      return;
    } else {
      ind = next;
    }

  }

}

//! Sparse Accumulation with KDTree
//!
//! We allocate and initialize the accumulation buffer,
//! which stores the estimate for each local point.
//! The accumulation procedure spawns points randomly in
//! the domain and then descends them along the gradient
//! computed based on nearest neighbors until they leave
//! the domain. If the nearest point switches at any point,
//! then we accumulate to that value.
//!
//! \todo Replace Gradient Computation Method
//! \todo Allow for Changing Point Height from Erosion / Slope
//!
soil::buffer sparseacc(const soil::kdtree& kdtree, const soil::buffer& points, const soil::index& index, const size_t niter){

  std::cout<<"Launching Sparse Accumulation"<<std::endl;

  // Initialize Accumulation
  soil::buffer_t<float> acc(points.elem(), soil::GPU);
  _init_acc<<<block(acc.elem(), 1024), 1024>>>(acc);

  std::cout<<"Seeding Random Number Generator..."<<std::endl;

  // Initialize Random State
  const size_t N = 8192;
  soil::buffer_t<curandState> rand(N, soil::host_t::GPU);
  seed<<<block(N, 1024), 1024>>>(rand, 0, 0);
  
  // Sparse Descent Kernel:
  //  Launch N kernels...
  const auto index_t = index.as<soil::flat_t<2>>();
  const auto point_t = points.as<vec3>();

  std::cout<<"Descending Particles..."<<std::endl;
  
  for(int i = 0; i < niter; ++i){
    sparse_descend<<<block(N, 1024), 1024>>>(kdtree, point_t, acc, rand, N, index_t);
  }

  std::cout<<"Done"<<std::endl;

  return soil::buffer(acc);

}

} // end of namespace soil

#endif