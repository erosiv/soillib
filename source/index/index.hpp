#ifndef SOILLIB_INDEX
#define SOILLIB_INDEX

#include <soillib/index/flat.hpp>

/*
The idea is that we have an index type, that allow for converting
coordinates to and from pure indices. This should basically replace
the currently existing shape type.

I am not quite sure what the optimal configuration of static and
dynamic typic is yet. In any case, this should effectively replace
the current implementation of the basic and quad maps.

How layer maps will work will be determined in the future.

The basic idea is to convert position -> index!
Index can be used to lookup values. This should also work for layers?
*/

namespace soil {

template<typename F, typename... Args>
auto indexselect(const soil::dindex type, F lambda, Args&&... args){
  switch(type){
    case soil::FLAT1: return lambda.template operator()<soil::flat_t<1>>(std::forward<Args>(args)...);
    case soil::FLAT2: return lambda.template operator()<soil::flat_t<2>>(std::forward<Args>(args)...);
    case soil::FLAT3: return lambda.template operator()<soil::flat_t<3>>(std::forward<Args>(args)...);
    case soil::FLAT4: return lambda.template operator()<soil::flat_t<4>>(std::forward<Args>(args)...);
    default: throw std::invalid_argument("index not supported");
  }
}

// select the index type and blablabla do the thing...

struct index {

  template<size_t D>
  using vec_t = glm::vec<D, int>;

  index(){}
  index(const vec_t<2> vec){
    this->impl = new flat_t<2>(vec);
  }

  dindex type() const noexcept {
    return this->impl->type();
  }

  template<typename T> inline T& as() noexcept {
    return static_cast<T&>(*(this->impl));
  }

  template<typename T> inline const T& as() const noexcept {
    return static_cast<T&>(*(this->impl));
  }

  template<size_t D>
  bool oob(const vec_t<D> vec) const {
    return indexselect(impl->type(), [self=this, vec]<typename T>() -> bool {
      if constexpr(std::same_as<vec_t<D>, typename T::vec_t>){
        auto index = self->as<T>();
        return index.oob(vec);
      } else {
        throw std::invalid_argument("underlying index type does not accept this argument");
      }
    });
  }

  template<size_t D>
  size_t flatten(const vec_t<D> vec) const {
    return indexselect(impl->type(), [self=this, vec]<typename T>() -> size_t {
      if constexpr(std::same_as<vec_t<D>, typename T::vec_t>){
        auto index = self->as<T>();
        return index.flatten(vec);
      } else {
        throw std::invalid_argument("underlying index type does not accept this argument");
      }
    });
  }

  size_t dims() const {
    return indexselect(impl->type(), [self=this]<typename T>(){
      return self->as<T>().dims();
    });
  }
  
  size_t elem() const {
    return indexselect(impl->type(), [self=this]<typename T>(){
      return self->as<T>().elem();
    });
  }

  //! \todo eliminate this because it makes no sense
  size_t operator[](const size_t d) const {
    return indexselect(impl->type(), [self=this, d]<typename T>(){
      return self->as<T>().operator[](d);
    });
  }

  auto iter() const {
    if(impl->type() == soil::FLAT2){
      return this->as<soil::flat_t<2>>().iter();
    }
    else throw std::invalid_argument("can't retrieve this iterator yet");
  }

private:
  indexbase* impl;
};

} // end of namespace soil

#endif