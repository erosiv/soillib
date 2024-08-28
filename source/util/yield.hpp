#ifndef SOILLIB_UTIL_YIELD
#define SOILLIB_UTIL_YIELD

#include <coroutine>
#include <cstdint>
#include <exception>
#include <utility>

namespace soil {

//! yield is a basic co-routine + promise generator with
//! correct error handling and yielded value move-semanticts.
//!
//! Implementation Taken From:
//! https://en.cppreference.com/w/cpp/language/coroutines#co_yield
//!
template<typename T>
struct yield {

  // Type Definitions

  struct promise_type;  //!< Promise-Type Forward Declaration
  typedef T value_type; //!< Return (Yield)-Value Type

  //! Local Call-Handle Type
  using handle_type = std::coroutine_handle<promise_type>;

  struct promise_type { //!< Promise-Type Definition

    value_type value_;             //!< Local Yield-Value Cache
    std::exception_ptr exception_; //!< Local Exception Cache

    yield get_return_object() {
      return yield(handle_type::from_promise(*this));
    }

    std::suspend_always initial_suspend() { return {}; }
    std::suspend_always final_suspend() noexcept { return {}; }

    //! Cache exception on unknown exception
    void unhandled_exception() { exception_ = std::current_exception(); }

    //! Automatically handles template-instantiation for
    //! conversion from any type that co_yield is called with.
    template<std::convertible_to<T> From> // C++20 concept
    std::suspend_always yield_value(From &&from) {
      value_ = std::forward<From>(from); // Cache result locally
      return {};
    }
    void return_void() {}
  };

  handle_type h_; //!< Local Coroutine Handle Instance

  // Constructors / Destructors
  //  Important Note:
  //  Coroutines are opaque and will be destroyed
  //  upon move, so we have to define custom move
  //  constructors which make sure they stay alive.

  yield(handle_type h): h_(h) {}

  yield(yield &&rhs): h_{std::exchange(rhs.h_, nullptr)} {}

  ~yield() {
    if (h_)
      h_.destroy();
  }

  yield &operator=(yield &&rhs) {
    if (this == &rhs)
      return *this;
    if (h_)
      h_.destroy();
    h_ = std::exchange(rhs.h_, nullptr);
    return *this;
  }

  // Boolean Completeness and Call Operators

  //! The bool cast operator checks if the coroutine has been
  //! completed by calling the coroutine (if not yielded) and
  //! determining if the coroutine handle has terminated.
  explicit operator bool() {
    retrieve_value();
    return !h_.done();
  }

  //! The call operator calls the coroutine and moves the
  //! cached value out as the return value, resetting the
  //! has_value_ flag to retrieve the next value.
  T operator()() {
    retrieve_value();
    has_value_ = false;
    return std::move(h_.promise().value_);
  }

private:
  bool has_value_ = false;

  void retrieve_value() {
    if (has_value_) // skip if cached value exists
      return;
    h_();                        // actual coroutine call here!
    if (h_.promise().exception_) // handle exception
      std::rethrow_exception(h_.promise().exception_);
    has_value_ = true; // we now have a cached value
  }

public:
  //! Generic yield Iterator Generator
  //! A yield acts as a coroutine-based value generator.
  //! This allows a yield to trivially implement iterators
  //! in a generic fashion using the generator concept.
  //!
  //! If a sequence can be expressed through co_yields of a coroutine,
  //! then this offers a simple interface where the iterator state
  //! doesn't have to be stored explicitly but is abstracted away into
  //! the coroutine state with correct construction / destruction.
  //!
  struct iterator {

    explicit iterator(): terminated{true},
                         handle{NULL} {};

    explicit iterator(yield *handle): terminated{false},
                                      handle(std::forward<yield *>(handle)) {
      step();
    }

    // Bounds-Checking
    bool operator==(const iterator &rhs) {
      return (this->terminated == rhs.terminated);
    }

    bool operator!=(const iterator &rhs) {
      return !(*this == rhs);
    }

    iterator &operator++() { //!< Progress yield
      step();
      return *this;
    }

    value_type operator*() { //!< Move Cached Value
      return std::move(value);
    }

  private:
    yield *handle;    //!< yield Callback
    value_type value; //!< Cached Value
    bool terminated;  //!< Iteration Completed

    void step() {
      if (*handle)
        value = (*handle)();
      else
        terminated = true;
    }
  };

  iterator begin() { return iterator{this}; }
  iterator end() { return iterator{}; }
};

} // end of namespace soil

#endif