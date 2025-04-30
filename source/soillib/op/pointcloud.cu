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
#include <soillib/op/rbf.hpp>
#include <soillib/op/cu_common.cu>

#include <cukd/builder.h>
#include <cukd/knn.h>

namespace soil {

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

  soil::buffer_t<curandState> randStates(N, soil::host_t::GPU);
  seed(randStates, 0, 0);

  soil::buffer_t<vec2> output(N, soil::GPU);
  _sample_N<<<block(N, 1024), 1024>>>(output, randStates, index, N);

  return output;

}

//
// Halton Sampler
//

__device__ float halton_val(int n, const int b){

  float f = 1.0f;
  float r = 0.0f;

  while(n > 0){

    f = f / b;
    r = r + f * (n % b);
    n = floorf(n / b);

  }

  return r;

}

__global__ void _sample_halton(soil::buffer_t<vec2> output, const soil::flat_t<2> index, const size_t N){

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= N) return;

  const float x = halton_val(n, 2) * (index[0] - 1);
  const float y = halton_val(n, 3) * (index[1] - 1);
  output[n] = soil::vec2(x, y);

}

soil::buffer_t<vec2> sample_halton_impl(const soil::flat_t<2> &index, const size_t N){

  soil::buffer_t<vec2> output{N, soil::GPU};
  _sample_halton<<<block(N, 1024), 1024>>>(output, index, N);
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
// Sample Gradient Implementation
//

__global__ void _sample_grad(const soil::buffer_t<float> field, soil::buffer_t<vec3> output, const soil::flat_t<2> index, const soil::buffer_t<vec2> pos_b){

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= pos_b.elem()) return;

  vec2 pos = pos_b[n];
  
  lerp5_t<float> lerp;
  lerp.gather(field, index, ivec2(pos));
  const vec2 grad = lerp.grad();
  const vec3 normal = glm::normalize(vec3(-grad.x, -grad.y, 1.0f));
  output[n] = normal;

}

soil::buffer_t<vec3> sample_grad_impl(const soil::buffer_t<float> &field, const soil::flat_t<2> &index, const soil::buffer_t<vec2>& pos){

  const size_t elem = pos.elem();
  soil::buffer_t<vec3> output(elem, soil::GPU);
  _sample_grad<<<block(elem, 1024), 1024>>>(field, output, index, pos);
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
  acc[n] = 1.0f;

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
__device__ void knn(const soil::kdtree& kdtree, const vec2 pos, cukd::HeapCandidateList<K>& list) {

  cukd::stackBased::knn<
    cukd::HeapCandidateList<K>,
    payload_t,
    payload_traits
  >(list, make_point(pos), kdtree.data.data(), kdtree.elem());

}

__device__ int nn(const soil::kdtree& kdtree, const vec2 pos) {
  return cukd::stackBased::fcp<
    payload_t,
    payload_traits
  >(make_point(pos), kdtree.data.data(), kdtree.elem());
}

//! Simple Multi-Dimensional Monomial Function
__device__ vec2 monomial_grad(const size_t o, const vec2 p){
  if(o == 0){ return vec2(0.0f); }
  if(o == 1){ return vec2(1.0f, 0.0f); }
  if(o == 2){ return vec2(0.0f, 1.0f); }
  if(o == 3){ return vec2(2.0f*p.x, 0.0f); }
  if(o == 4){ return vec2(0.0f, 2.0f*p.y); }
  if(o == 5){ return vec2(p.y, p.x); }
  return vec2(0.0f);
}

//! Greedy Descent: Steepest Slope from Neighbors
__global__ void sparse_descend(
  const soil::rbf rbf,
  const soil::kdtree kdtree,
  soil::buffer_t<float> acc,
  soil::buffer_t<curandState> rand,
  const size_t N,
  const soil::flat_t<2> index,
  const float prob) {

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= N) return;

  // Initialize Position in Domain
  int ind = -1;
  curandState* randState = &rand[n];
  vec2 pos = vec2(
    curand_uniform(randState)*(index[0]-1),
    curand_uniform(randState)*(index[1]-1)
  );

  const size_t K = rbf.elem();  //!< Total Points
  const size_t B = 64;          //!< Nearest Points
  const size_t P = rbf.P;       //!< Monomial Terms
  
  const float rad = rbf.shape * 10.0f;
  cukd::HeapCandidateList<B> list(rad);

  int maxstep = 256;
  while(maxstep > 0){
    maxstep--;

    knn<B>(kdtree, pos, list);
    
    // Radial Basis Set Gradient
    vec2 grad = vec2(0.0f);
    for(int b = 0; b < B; ++b){

      int k = list.get_pointID(b);
      if(k >= 0){
        k = kdtree.data[k].i;

        const vec2 c = rbf.centers[k];
        const float w = rbf.weights[k];
        const float s = rbf.shape;
        const vec2 d = (pos - c)/s;
        const float r = glm::length(c - pos);
        grad -= d * w * 2.0f * rbf::func(r / s) * r / s / s;

      }

    }
    
    // Monomial Basis Set Gradient
    for(int p = 0; p < P; ++p){
      const float w = rbf.weights[K + p];
      grad += w * monomial_grad(p, pos);
    }

    pos -= rbf.shape * glm::normalize(grad);

    int nearest = nn(kdtree, pos);
    if(nearest >= 0){
      nearest = kdtree.data[nearest].i;
      if(nearest != ind){
        ind = nearest;
        atomicAdd(&acc[ind], prob);
      }
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
soil::buffer sparseacc(const soil::rbf& rbf, const soil::kdtree& kdtree, const soil::index& index, const size_t niter){

  std::cout<<"Launching Sparse Accumulation"<<std::endl;

  // Initialize Accumulation
  soil::buffer_t<float> acc(rbf.elem(), soil::GPU);
  _init_acc<<<block(acc.elem(), 1024), 1024>>>(acc);

  // Initialize Random State
  std::cout<<"Seeding Random Number Generator..."<<std::endl;
  const size_t N = 2048;
  soil::buffer_t<curandState> randStates(N, soil::host_t::GPU);
  seed(randStates, 0, 0);
  
  // normalization factor ...
  const float Ntot = niter * N;             //!< Total Sample Count
  const float P = float(rbf.elem()) / Ntot; //!< Total Probability

  // Sparse Descent Kernel:
  //  Launch N kernels...
  const auto index_t = index.as<soil::flat_t<2>>();
  std::cout<<"Descending Particles..."<<std::endl;
  for(int i = 0; i < niter; ++i){
    std::cout<<"Iteration "<<i<<std::endl;
    sparse_descend<<<block(N, 128), 128>>>(rbf, kdtree, acc, randStates, N, index_t, P);
    cudaDeviceSynchronize();
  }

  return soil::buffer(acc);

}

} // end of namespace soil

#endif