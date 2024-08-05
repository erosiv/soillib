#ifndef SOILLIB_TIMER
#define SOILLIB_TIMER

#include <chrono>
//#include <iostream>

namespace soil {

//template<typename D>
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