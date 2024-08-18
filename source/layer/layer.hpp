#ifndef SOILLIB_LAYER
#define SOILLIB_LAYER

#include <soillib/soillib.hpp>
#include <soillib/util/types.hpp>

#include <soillib/layer/constant.hpp>
#include <soillib/layer/stored.hpp>



/*
Layer Concept:

Like in GIMP, or Photoshop, a full model consists of multiple layers of information.
This name might be subject to change, since they are not necessarily "vertically layered".

The idea is that these act as generic maps, which take the data from an array or another
spatially organizing structure and allow for computing maps directly, caching results and
returning fully pre-computed arrays on request.

We will try to implement this for now and later refine the concept to be more modular.
This can later also be used to have a node-based synthesis model, but primarily exists
so that we can define common operations and even one day autograd through them.
*/

/*
Implementation: Does the layer have a strict-typed input and output? I think so.
Does it store a reference to its underlying in type?

In theory, I could have a generic node that just takes a lambda right?
Let's not do that for now though. In theory this could be done later
for proper interactive nodes w. std::function?
*/
namespace {

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