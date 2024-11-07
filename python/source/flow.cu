#include <soillib/node/algorithm/flow_test.hpp>
#include <cuda_runtime.h>
#include <iostream>

void soil::flow_test::test() const {

  std::cout<<"CUDA LINKED"<<std::endl;

  /*
  // execute some cuda code.
  const size_t elem = index.elem();
  auto in = this->buffer.as<int>();
  auto out = buffer_t<ivec2>{elem};

  for(size_t i = 0; i < elem; ++i){
    glm::ivec2 val(0, 0);
    for(size_t k = 0; k < 8; ++k){
      if(in[i] == dirmap[k]){
        val = coords[k];
        break;
      }
    }
    out[i] = val;
  }

  return std::move(soil::buffer(std::move(out)));
  */
}