#include <soillib/soillib.hpp>

#include <soillib/util/newbuf.hpp>

#include <iostream>

int main(int argc, char *args[]) {

  soil::nn::buf buf_f("float");
  buf_f.allocate(32);  
  std::cout<<buf_f.size()<<std::endl;

  soil::nn::buf buf_d("double");
  buf_d.allocate(32);
  std::cout<<buf_d.size()<<std::endl;

}
