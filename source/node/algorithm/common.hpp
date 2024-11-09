#ifndef SOILLIB_NODE_ALGORITHM_COMMON
#define SOILLIB_NODE_ALGORITHM_COMMON

#include <soillib/core/buffer.hpp>

namespace soil {

template<typename T>
void fill(soil::buffer_t<T>& buffer, const T val){
  // ... fill the guy ...
}

// in-place addition? out-of-place addition?
void add(soil::buffer& lhs, soil::buffer& rhs){
}

void subtract(soil::buffer& lhs, soil::buffer& rhs){
}

void multiply(soil::buffer& lhs, soil::buffer& rhs){
}




/*
-> Add
-> Multiply
-> Something else? A generic way to do this with methods?
  -> We can probably make it efficient using templates.
-> 

Add dlpack interop w. torch

- Add more generic operations so that buffers are easy to manipulate

*/

}

#endif
