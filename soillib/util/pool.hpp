#ifndef SOILLIB_UTIL_POOL
#define SOILLIB_UTIL_POOL

#include <soillib/soillib.hpp>
#include <soillib/util/buf.hpp>

#include <stdexcept>
#include <deque>

namespace soil {

/*************************************************
pool is a basic fixed-size allocation memory pool
for retrieving individual elements through in-place
construction.

todo:
- use crtp to register the overall size of all
  pools, for static estimation of whether or not
  the program will run out of memory. maybe?
- add a set sorting to make fewer fragmented data
  segments
*************************************************/

template<typename T>
struct pool {

  soil::buf<T> root;          // Owning Memory Segment
  std::deque<T*> free;         // Reference to Available Memory

  pool(){}
  pool(size_t size):root(size){
    for(size_t i = 0; i < root.size; ++i)
      free.emplace_front(root.start + i);
  }

  // Data Retrieval and Returning

  template<typename... Args>
  T* get(Args && ...args){ // In-Place Construction

    if(free.empty()){
      throw std::invalid_argument( "out of memory" );
      return NULL;
    }

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
