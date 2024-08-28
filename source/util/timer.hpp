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
struct timer {

  // Construction w. Duration Specifier

  enum duration {
    SECONDS,
    MILLISECONDS,
    MICROSECONDS,
    NANOSECONDS
  };

  timer(const duration d): d{d} {}

  timer(): timer(duration::MILLISECONDS) {}

  // Start / Stop / Count Implementation

  void start() {
    this->_start = std::chrono::high_resolution_clock::now();
  }

  void stop() {
    this->_stop = std::chrono::high_resolution_clock::now();
  }

  size_t count(duration d) const {
    switch (d) {
    case SECONDS:
      return count_t<std::chrono::seconds>();
    case MILLISECONDS:
      return count_t<std::chrono::milliseconds>();
    case MICROSECONDS:
      return count_t<std::chrono::microseconds>();
    case NANOSECONDS:
      return count_t<std::chrono::nanoseconds>();
    default:
      throw std::invalid_argument("unrecognized duration enumerator");
    }
  }

  size_t count() const {
    return count(this->d);
  }

private:
  template<typename D>
  size_t count_t() const {
    auto duration = std::chrono::duration_cast<D>(this->_stop - this->_start);
    return duration.count();
  }

  duration d;
  std::chrono::time_point<std::chrono::high_resolution_clock> _start;
  std::chrono::time_point<std::chrono::high_resolution_clock> _stop;
};

} // end of namespace soil

#endif