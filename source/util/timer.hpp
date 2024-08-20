#ifndef SOILLIB_UTIL_TIMER
#define SOILLIB_UTIL_TIMER

#include <chrono>

namespace soil {

//! timer is a simple struct that wraps around
//! std::chrono high-resolution clock to provide
//! a benchmarking method.
//!
//! Exposed to python through the `with` operator,
//! this provides a simple scope-based benchmark.
//!
//! \todo Add an option to specify the time unit.
//!   -> with soil.timer('ms') as timer:
//!
struct timer {

  void start(){
    this->_start = std::chrono::high_resolution_clock::now();
  }

  void stop(){
    this->_stop = std::chrono::high_resolution_clock::now();
  }

  template<typename D = std::chrono::milliseconds>
  size_t count(){
    auto duration = std::chrono::duration_cast<D>(this->_stop - this->_start);
    return duration.count();
  }

private: 
  std::chrono::time_point<std::chrono::high_resolution_clock> _start;
  std::chrono::time_point<std::chrono::high_resolution_clock> _stop;
};

} // end of namespace soil

#endif