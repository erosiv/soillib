#ifndef SOILLIB_UTIL_YIELD
#define SOILLIB_UTIL_YIELD

#include <coroutine>
#include <cstdint>
#include <exception>
#include <utility>
#include <type_traits>

namespace soil {

//! yield_t is a basic co-routine + promise generator with
//! correct error handling and yielded value move-semanticts.
//!
//! yield_t is strict-typed, templated on a single yielded type.
//! Multiple yield values should be packed in tuple, for easy
//! unpacking in range based for loop expressions.
//!
//! Convenience construction templates are provided for this purpose.
//!
//! Implementation Taken From:
//! https://en.cppreference.com/w/cpp/language/coroutines#co_yield_t
//!
template<typename T>
struct yield_t;

//! \todo Fix this to allow for reference passing.
//! That would ultimately require more template meta-programming,
//! to remove the references, convert the tuple packto a pointer,
//! with correct construction from make_yield (and direct construction),
//! and finally making it a reference again when dereferencing the iterator.

//
// Convenience Interface
//

namespace {

//! yield_v is a helper template for packing multiple values into a tuple.
template<typename... Args>
struct yield_v {
  static_assert(!(std::is_reference_v<Args> || ...), "references are not permitted in yield return values. use a pointer instead");
  typedef std::tuple<Args...> value_t; 
};

template<typename T>
struct yield_v<T> {
  static_assert(!std::is_reference_v<T>, "references are not permitted in yield return values. use a pointer instead");
  typedef T value_t; 
};

}

//! make_yield is a convenient constructor function for building
//! the correct return value type for a given yield.
//!
//! If the values passed to the method have the correct strict-type,
//! then the correct yield value type is constructed directly, and
//! no template parameters have to be specified.
//!
template<typename... Args>
yield_v<Args...>::value_t make_yield(Args... args){
  return typename yield_v<Args...>::value_t(std::forward<Args>(args)...);
}

//! yield is the primary interface for building yielding iterators.
//!
//! It is a convenient alias of yield_t, which packs multiple
//! template parameters into a single tuple return type. 
//!
//! Usage:
//!
//! yield<size_t> my_iterator() const {
//!   for(size_t i = 0; i < this->size(); ++i)
//!     co_yield i;
//!   co_return;
//! }
//!
//! for(size_t index: my_iterator())
//!   std::print(index);
//! 
//! yield<size_t, T> my_iterator() const {
//!   for(size_t i = 0; i < this->size(); ++i)
//!     co_yield make_yield(i, this->operator[](i));
//!   co_return;
//! }
//!
//! for([index, value]: my_iterator())
//!   std::print(index, value);
//!
template<typename... Args>
using yield = yield_t<typename yield_v<Args...>::value_t>;

//
// Implementation
//

template<typename T>
struct yield_t {

  // Type Definitions

  struct promise_type;  //!< Promise-Type Forward Declaration
  typedef T value_type; //!< Return (yield_t)-Value Type

  //! Local Call-Handle Type
  using handle_type = std::coroutine_handle<promise_type>;

  struct promise_type { //!< Promise-Type Definition

    value_type value_;             //!< Local yield_t-Value Cache
    std::exception_ptr exception_; //!< Local Exception Cache

    yield_t get_return_object() {
      return yield_t(handle_type::from_promise(*this));
    }

    std::suspend_always initial_suspend() { return {}; }
    std::suspend_always final_suspend() noexcept { return {}; }

    //! Cache exception on unknown exception
    void unhandled_exception() { exception_ = std::current_exception(); }

    //! Automatically handles template-instantiation for
    //! conversion from any type that co_yield_t is called with.
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

  yield_t(handle_type h): h_(h) {}

  yield_t(yield_t &&rhs): h_{std::exchange(rhs.h_, nullptr)} {}

  ~yield_t() {
    if (h_)
      h_.destroy();
  }

  yield_t &operator=(yield_t &&rhs) {
    if (this == &rhs)
      return *this;
    if (h_)
      h_.destroy();
    h_ = std::exchange(rhs.h_, nullptr);
    return *this;
  }

  // Boolean Completeness and Call Operators

  //! The bool cast operator checks if the coroutine has been
  //! completed by calling the coroutine (if not yield_ted) and
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
  //! Generic yield_t Iterator Generator
  //! A yield_t acts as a coroutine-based value generator.
  //! This allows a yield_t to trivially implement iterators
  //! in a generic fashion using the generator concept.
  //!
  //! If a sequence can be expressed through co_yield_ts of a coroutine,
  //! then this offers a simple interface where the iterator state
  //! doesn't have to be stored explicitly but is abstracted away into
  //! the coroutine state with correct construction / destruction.
  //!
  struct iterator {

    explicit iterator(): terminated{true},
                         handle{NULL} {};

    explicit iterator(yield_t *handle): terminated{false},
                                      handle(std::forward<yield_t *>(handle)) {
      step();
    }

    // Bounds-Checking
    bool operator==(const iterator &rhs) {
      return (this->terminated == rhs.terminated);
    }

    bool operator!=(const iterator &rhs) {
      return !(*this == rhs);
    }

    iterator &operator++() { //!< Progress yield_t
      step();
      return *this;
    }

    value_type operator*() { //!< Move Cached Value
      return std::move(value);
    }

  private:
    yield_t *handle;  //!< yield_t Callback
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