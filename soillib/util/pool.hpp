#ifndef SOILLIB_UTIL_POOL
#define SOILLIB_UTIL_POOL

#include <soillib/soillib.hpp>
#include <soillib/util/buf.hpp>

#include <deque>

namespace soil {

/*************************************************
pool is a basic fixed-size allocation memory pool
for retrieving sub-slices of larger pools, as well
as in place construction and data returning.

todo:
- use crtp to register the overall size of all
  pools, for static estimation of whether or not
  the program will run out of memory.
- add a set sorting to make fewer fragmented data
  segments
- add in-place single element construct and get
*************************************************/

template<typename T>
struct pool {

  soil::buf<T> root;          // Owning Memory Segment
  std::deque<buf<T>> free;    // Reference to Available Memory

  pool(){}
  pool(size_t _size){
    reserve(_size);
  }

  ~pool(){
    clear();
  }

  // Bulk Manipulations

  void reserve(size_t _size){
    root.size = _size;
    root.start = new T[root.size];
    free.emplace_front(root.start, root.size);
  }

  void clear(){
    free.clear();
    if(root.start != NULL){
      delete[] root.start;
      root.start = NULL;
      root.size = 0;
    }
  }

  // Data Retrieval and Returning

  buf<T> get(size_t _size){

    if(free.empty())
      return {NULL, 0};

    if(_size > root.size)
      return {NULL, 0};

    if(free.front().size < _size)
      return {NULL, 0};

    buf<T> sec = {free.front().start, _size};
    free.front().start += _size;
    free.front().size -= _size;

    return sec;

  }

  /*

    //Return Element
    void unget(sec* E){
      if(E == NULL)
        return;
      E->reset();
      free.push_front(E);
    }

    void reset(){
      free.clear();
      for(int i = 0; i < size; i++)
        free.push_front(start+i);
    }


      //Retrieve Element, Construct in Place
      template<typename... Args>
      sec* get(Args && ...args){

        if(free.empty()){
          cout<<"Memory Pool Out-Of-Elements"<<endl;
          return NULL;
        }

        sec* E = free.back();
        try{ new (E)sec(forward<Args>(args)...); }
        catch(...) { throw; }
        free.pop_back();
        return E;

      }

    */

};

}; // end of namespace

#endif
