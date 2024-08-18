#ifndef SOILLIB_LAYER
#define SOILLIB_LAYER

#include <soillib/soillib.hpp>
#include <soillib/util/types.hpp>

// #include <soillib/layer/stored.hpp>

namespace soil {

//! A layer represents constant, stored or computed
//! quantity distributed over the domain.
//!
//! Layers are modular and can be composed to yield
//! user-defined quantities, which are necessary for
//! executing any dynamic transport model.
//! 
//! Layers can also cache and pre-compute results for
//! efficient deferred computation.

template<typename T>
struct layer_t {

  //! Compute the Output Quantity
  virtual T operator()(const size_t index) noexcept;

};

/*
enum TEST {
  A, B
};

struct test_base{};

template<TEST T>
struct my_struct: test_base {
  int a;
};

test_base make_test(const TEST my_test){
  switch(my_test){
    case A: return my_struct<A>{};
    case B: return my_struct<B>{};
    default: throw std::invalid_argument("INVALID ARGUMENT");
  }
}

}






template<typename T>
concept test_t = requires(T t){
  { t(size_t()) } -> std::convertible_to<soil::multi>;
};

namespace soil {

using layer_v = std::variant<
  soil::constant,
  soil::stored
>;

struct layer {

  layer(){}
  layer(layer_v&& _layer):
    _layer{_layer}{}

  soil::multi operator()(const size_t& index) const {
    return std::visit(overloaded {
      [](test_t t){ }
    });
    return std::visit([&index](auto&& args) -> soil::multi {
      return args(index);
    }, this->_constant);
  }




private:
  layer_v _layer;
};

*/


} // end of namespace soil

#endif