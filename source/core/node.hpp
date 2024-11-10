#ifndef SOILLIBnode
#define SOILLIBnode

#include <soillib/core/index.hpp>
#include <soillib/core/types.hpp>
#include <soillib/core/buffer.hpp>
#include <soillib/soillib.hpp>

#include <soillib/util/error.hpp>
#include <functional>

namespace soil {

//! node represents a deferred, cachable polymorphic computation graph.
//!
//! Nodes can represent any mapping from any set of value types to
//! any other value type, and store references to dependencies so
//! that the computation can be deferred, cached and manipulated.
//!
//! This has the advantage that complex models can be constructed
//! at run-time by node composition, and can be passed to the back-end
//! without require re-compilation. The computations themselves are
//! still strict-typed, while shape and type constraints are checked
//! at construction.
//!
//! A node can have data as its input dependency, allowing for the
//! definition of dynamically computed quantities which are dependent
//! on other sources of data.
//!
//! Ultimately, we receive an efficient computation graph that can be
//! flexibly parameterized, manipulated and extended.
//!
//! In the future, this can be used together with autograd concepts.
struct node;

//! Nodes are constructed using mapping functions. 
//! - Applied to a node, the computation is deferred and a node is returned.
//! - Applied to a buffer, the computation is immediate and a buffer is returned.
struct map;

//! map_t is a strict-typed map for transforming data.
//! 
//! 
template<typename T>
struct map_t: typedbase {

  using param_t = std::vector<node>;
  using func_t = std::function<T(const param_t&, const size_t)>;
  using rfunc_t = std::function<T&(const param_t&, const size_t)>;

  map_t(func_t func): func(func) {}
  map_t(func_t func, rfunc_t rfunc): func(func), rfunc{rfunc} {}

  constexpr soil::dtype type() noexcept override {
    return soil::typedesc<T>::type;
  }

  T val(const param_t& param, const size_t index) const noexcept {
    return this->func(param, index);
  }

  T& ref(const param_t& param, const size_t index) noexcept {
    return this->rfunc(param, index);
  }

private:
  func_t func;
  rfunc_t rfunc = [](const param_t& param, const size_t index) -> T& {
    throw std::runtime_error("NO BACKPROPAGATION POSSIBLE");
  };
};

struct map {

  using param_t = std::vector<node>;

  template<typename T>
  using func_t = std::function<T(const param_t&, const size_t)>;

  template<typename T>
  using rfunc_t = std::function<T&(const param_t&, const size_t)>;

  map() {}

  template<typename T>
  map(func_t<T> func): impl{make<T>(func)}{}

  template<typename T>
  map(func_t<T> func, rfunc_t<T> rfunc): impl{make<T>(func, rfunc)}{}

  inline soil::dtype type() const noexcept {
    return this->impl->type();
  }

  // Strict Type Casting

  template<typename T>
  inline const map_t<T>& as() const noexcept {
    return static_cast<map_t<T> &>(*(this->impl));
  }

  template<typename T>
  inline map_t<T>& as() noexcept {
    return static_cast<map_t<T> &>(*(this->impl));
  }

  // Value and Reference Retrieval

  //! Sample Value
  template<typename T>
  T val(std::vector<node>& in, const size_t index) const {
    return this->as<T>().val(in, index);
  }

  //! Retrieve Value Reference
  template<typename T>
  T& ref(std::vector<node>& in, const size_t index){
    return this->as<T>().ref(in, index);
  }

private:
  using ptr_t = std::shared_ptr<typedbase>;
  ptr_t impl; //!< Strict-Typed Implementation Base Pointer

  template<typename T>
  static ptr_t make(func_t<T> func) {
    return std::make_shared<soil::map_t<T>>(func);
  }

  template<typename T>
  static ptr_t make(func_t<T> func, rfunc_t<T> rfunc) {
    return std::make_shared<soil::map_t<T>>(func, rfunc);
  }
};

struct node {

  //! \todo Validity Checks on Construction
  //! \todo Figure out how to add parameters to map
  //! \todo Figure out if the params are necessary at all
  //! \todo Decide where full buffer conversion should happen
  //! \todo Figure out what happens with buffer parameters.
  //! \todo Figure out if shape is relevant at all...

  node(const soil::node& node){
    this->map = node.map;
    this->in = node.in;
    this->size = node.size;
  }

  node(soil::map map): map{map}{}
  node(soil::map map, std::vector<node> in): map{map}, in{in}{}

  // Value and Reference Retrieval from Map

  //! Sample Value
  template<typename T>
  T val(const size_t index) const {
    return this->map.as<T>().val(this->in, index);
  }

  //! Retrieve Value Reference
  template<typename T>
  T& ref(const size_t index){
    return this->map.as<T>().ref(this->in, index);
  }

  // template<typename T>
  // soil::buffer_t<T> operator()() {
  //   return this->map.template operator()<T>(this->in);
  // }

  //! Retrieve Node Return Value Type
  inline soil::dtype type() const noexcept {
    return this->map.type();
  }

private:
  std::vector<node> in;
  soil::map map;
public:
  size_t size = 0;
};

} // end of namespace soil

#endif