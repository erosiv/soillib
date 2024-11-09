#ifndef SOILLIB_ERROR
#define SOILLIB_ERROR

#include <soillib/core/types.hpp>
#include <soillib/soillib.hpp>
#include <sstream>

// Custom Soillib Exceptions

namespace soil {
namespace error {

template<typename From, typename To>
struct cast_error: std::exception {
  static std::string value() noexcept {
    return std::format("invalid cast from <{}> to <{}>", typedesc<From>::name, typedesc<To>::name);
  }
  const char *what() const noexcept override {
    return value().c_str();
  }
};

template<typename Type>
struct type_op_error: std::exception {
  static std::string value() noexcept {
    return std::format("invalid type <{}> for operation", typedesc<Type>::name);
  }
  const char *what() const noexcept override {
    return value().c_str();
  }
};

// Mismatch Errors

struct mismatch_size: std::exception {
  mismatch_size(size_t lhs, size_t rhs){
    std::stringstream ss;
    ss << "mismatch size. want lhs(";
    ss << lhs;
    ss << "), have rhs(";
    ss << rhs;
    ss << ")";
    this->msg = ss.str();
  }
  const char *what() const noexcept override {
    return this->msg.c_str();
  }
private:
  std::string msg;
};

struct mismatch_type: std::exception {
  mismatch_type(dtype lhs, dtype rhs){
    std::stringstream ss;
    ss << "mismatch type. want lhs(";
    ss << soil::select(lhs, []<typename S>(){ return typedesc<S>::name; });
    ss << "), have rhs(";
    ss << soil::select(rhs, []<typename S>(){ return typedesc<S>::name; });
    ss << ")";
    this->msg = ss.str();
  }
  const char *what() const noexcept override {
    return this->msg.c_str();
  }
private:
  std::string msg;
};

struct mismatch_host: std::exception {
  mismatch_host(host_t lhs, host_t rhs){
    std::stringstream ss;
    ss << "mismatch host. want lhs(";
    ss << soil::select(lhs, []<host_t S>(){ return hostdesc<S>::name; });
    ss << "), have rhs(";
    ss << soil::select(rhs, []<host_t S>(){ return hostdesc<S>::name; });
    ss << ")";
    this->msg = ss.str();
  }
  const char *what() const noexcept override {
    return this->msg.c_str();
  }
private:
  std::string msg;
};

} // end of namespace error
} // end of namespace soil

#endif