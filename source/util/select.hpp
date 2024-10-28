#ifndef SOILLIB_SELECT
#define SOILLIB_SELECT

#include <cstddef>

namespace soil {

/*
The goal is to have a select function,
which lets us get the template parameters
that we need... based on an enumerator?

Or a set of enumerators ideally.
*/

// Set of Enumerators

enum EnumA: size_t;
enum EnumB: size_t;

// We want to be able to VALUE pass
// the enumerator to a function, which
// calls a strict-type lambda!

template<EnumA E>
struct getType;

enum EnumA: size_t {
  A, B, C
};

template <> struct getType<EnumA::A>  { using type = float; };
// template <> struct getType<DOUBLE> { using type = double; };
// template <> struct getType<INT>    { using type = int; };








template<typename E, typename F, typename ...Args>
auto _select(const E e, F lambda, Args&& ...args){
  switch(e){
    case 0: 
      using type = getType<soil::EnumA{0}>::type;
      return lambda.template operator()<type>(std::forward<Args>(args)...);
    default:
      throw std::runtime_error("unexpected value");
  }
///  return 

  // here we should theoretically switch...
  // but that would require knowing all components of the
  // enumerator ahead of time. or some const way to convert
  // the enumerator to a decltype...
}

// Implementation



/*

template<typename F, typename... Args>
auto select(const soil::dtype type, F lambda, Args &&...args) {
  switch (type) {
  case soil::INT:
    return lambda.template operator()<int>(std::forward<Args>(args)...);
  case soil::FLOAT32:
    return lambda.template operator()<float>(std::forward<Args>(args)...);
  case soil::FLOAT64:
    return lambda.template operator()<double>(std::forward<Args>(args)...);
  case soil::VEC2:
    return lambda.template operator()<vec2>(std::forward<Args>(args)...);
  case soil::VEC3:
    return lambda.template operator()<vec3>(std::forward<Args>(args)...);
  default:
    throw std::invalid_argument("type not supported");
  }
}
*/
/*

enum TestEnum {
  A, B, C
};

template<TestEnum T>
void test(){
//  std::cout<<T<<std::endl;
}

template<typename Enum>
auto select(Enum e){
  return test<e>();

};
*/

} // end of namespace soil

#endif