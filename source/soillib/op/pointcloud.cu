#ifndef SOILLIB_OP_POINTCLOUD
#define SOILLIB_OP_POINTCLOUD
#define HAS_CUDA

#include <soillib/soillib.hpp>
#include <soillib/core/buffer.hpp>
#include <soillib/core/index.hpp>

#include <soillib/op/common.hpp>

namespace soil {

soil::buffer_t<vec3> sample_pointcloud_impl(const soil::buffer_t<float> &buffer, const soil::index &index, const size_t N){

  soil::buffer_t<vec3> output(N, soil::GPU);
  soil::set(output, vec3(0.0f));
  return output;

}

/*
template<typename T, typename Index, typename Flat>
__global__ void _resample(soil::buffer_t<T> input, soil::buffer_t<T> output, const Index index, const Flat flat){
  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= flat.elem()) return;
  
  auto pos = flat.unflatten(n);
  if(!index.oob(pos)){
    output[n] = input[index.flatten(pos)];
  }
}

template<typename T>
soil::buffer_t<T> resample_impl(const soil::buffer_t<T>& input, const soil::index& index){
  
return select(index.type(), [&]<typename I>(){
  
auto index_t = index.as<I>();
soil::flat_t<I::n_dims> flat(index_t.ext());

soil::buffer_t<T> output(flat.elem(), soil::GPU);
using V = soil::typedesc<T>::value_t;
T value = T{std::numeric_limits<V>::quiet_NaN()};
set_impl<T>(output, value, 0, flat.elem(), 1);

int thread = 1024;
int elem = flat.elem();
int block = (elem + thread - 1)/thread;
_resample<<<block, thread>>>(input, output, index_t, flat);

return output;

});

}
*/


}

#endif