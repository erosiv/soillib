#ifndef SOILLIB_ERROR
#define SOILLIB_ERROR

#include <soillib/core/types.hpp>
#include <soillib/soillib.hpp>
#include <sstream>

// Custom Soillib Exceptions

namespace soil {
namespace error {

#ifdef HAS_CUDA
#define gpuErrchk(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort)
      exit(code);
  }
}
#endif

template<typename From, typename To>
struct cast_error: std::exception {
  static std::string value() noexcept {
    return std::format("invalid cast from <{}> to <{}>", typedesc<From>::name, typedesc<To>::name);
  }
  const char *what() const noexcept override {
    return value().c_str();
  }
};

// Mismatch Errors

struct mismatch_size: std::exception {
  mismatch_size(size_t lhs, size_t rhs) {
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
  mismatch_type(dtype lhs, dtype rhs) {
    std::stringstream ss;
    ss << "mismatch type. want lhs(";
    ss << soil::select(lhs, []<typename S>() { return typedesc<S>::name; });
    ss << "), have rhs(";
    ss << soil::select(rhs, []<typename S>() { return typedesc<S>::name; });
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
  mismatch_host(host_t lhs, host_t rhs) {
    std::stringstream ss;
    ss << "mismatched host. want lhs<";
    ss << soil::select(lhs, []<host_t S>() { return hostdesc<S>::name; });
    ss << ">, have rhs<";
    ss << soil::select(rhs, []<host_t S>() { return hostdesc<S>::name; });
    ss << ">";
    this->msg = ss.str();
  }
  const char *what() const noexcept override {
    return this->msg.c_str();
  }

private:
  std::string msg;
};

struct unsupported_host: std::exception {
  unsupported_host(host_t lhs, host_t rhs) {
    std::stringstream ss;
    ss << "operation not support for host. want <";
    ss << soil::select(lhs, []<host_t S>() { return hostdesc<S>::name; });
    ss << ">, have <";
    ss << soil::select(rhs, []<host_t S>() { return hostdesc<S>::name; });
    ss << ">";
    this->msg = ss.str();
  }
  const char *what() const noexcept override {
    return this->msg.c_str();
  }

private:
  std::string msg;
};

struct missing_file: std::exception {
  missing_file(std::string file) {
    std::stringstream ss;
    ss << "File does not exist. ";
    ss << file;
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