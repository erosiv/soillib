#ifndef SOILLIB_INDEX_KDTREE_CU
#define SOILLIB_INDEX_KDTREE_CU
#define HAS_CUDA

#include <soillib/index/kdtree.hpp>
#include <cukd/builder.h>
#include <cukd/knn.h>

namespace soil {

void kdtree::deallocate(){
  if(this->data != NULL){
    cudaFree(this->data);
    this->data = NULL;
    this->_elem = 0;
  }
}

void kdtree::allocate(const size_t elem){
  cudaMalloc(&this->data, elem);
  this->_elem = elem;
}

void kdtree::setup(const buffer_t<vec3>& buffer){
  std::cout<<"Building Tree"<<std::endl;
  //  cukd::buildTree((float3*)buffer.data(), buffer.elem());
}

/*

//
// KDTree Implementation
//

namespace {

__global__ void _knnquery(const soil::buffer_t<vec3> query_b, const soil::buffer_t<vec3> data, soil::buffer_t<vec3> output, const size_t K){

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= query_b.elem()) return;

  const float3* data_ptr = (float3*)data.data();
  const size_t N = data.elem();
  const vec3 q = query_b[n];

  // Candidate List, Query Result
  cukd::HeapCandidateList<16> result(100.0);
  cukd::stackBased::knn(result, make_float3(q.x, q.y, q.z), data_ptr, N);

  for(int k = 0; k < K; ++k) {
    int ID = result.get_pointID(k);
    vec3 r = ID < 0
      ? vec3(0.f,0.f,0.f)
      : data[ID];
    output[n * K + k] = r;

  }

}

}

void knnbuild(soil::buffer_t<vec3>& buffer){

}

soil::buffer_t<vec3> knnquery(const soil::buffer_t<vec3>& data, const soil::buffer_t<vec3>& query, const size_t k) {

  const size_t n_query = query.elem()/3;
  auto output = soil::buffer_t<vec3>{k * n_query, soil::host_t::GPU};
  _knnquery<<<block(n_query, 512), 512>>>(query, data, output, k);
  return output;

}

*/

}

#endif