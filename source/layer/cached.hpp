#ifndef SOILLIB_LAYER_CACHED
#define SOILLIB_LAYER_CACHED

namespace soil {

template<typename T>
struct cached_t: typedbase, layer_t<T> {

  constant_t(const T value):
    value{value}{}

  constexpr soil::dtype type() const noexcept override { 
    return soil::typedesc<T>::type; 
  }

  T operator()(const size_t index) noexcept override {
    return this->value;
  }

private:
  const T value;
};

struct cached {};

} // end of namespace soil

#endif