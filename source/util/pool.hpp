#ifndef SOILLIB_UTIL_POOL
#define SOILLIB_UTIL_POOL

#include <soillib/soillib.hpp>
#include <soillib/util/buf.hpp>

#include <deque>

namespace soil {

//! pool<T> is a fixed-size allocation, owning memory pool,
//! for single-element retrieval and in-place construction.
//!
template<typename T>
struct pool {

  soil::buf<T> root;    //!< Memory Buffer (Owning)
  std::deque<T*> free;  //!< Available Memory Reference

  pool(){}
  pool(size_t size):root(size){
    for(size_t i = 0; i < root.size; ++i)
      free.emplace_front(root.start + i);
  }

  // Data Retrieval and Returning

  template<typename... Args>
  T* get(Args && ...args){ // In-Place Construction

    if(free.empty())
      return NULL;

    T* E = free.back();
    try{ new (E)T(std::forward<Args>(args)...); }
    catch(...) { throw; }
    free.pop_back();
    return E;

  }

  void unget(T* E){
    if(E == NULL)
      return;
    if(E >= root.start + root.size)
      return;
    free.push_front(E);
  }

};

}; // end of namespace

#endif
