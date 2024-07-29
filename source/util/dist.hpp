#ifndef SOILLIB_UTIL_DIST
#define SOILLIB_UTIL_DIST

#include <soillib/soillib.hpp>
#include <random>

namespace soil {
namespace dist {

std::random_device rd;
std::mt19937 gen(rd());

void seed(int seed){
  gen.seed(seed);
}

//Base Distributions

std::bernoulli_distribution brn(0.5);
bool bernoulli(){
  return brn(gen);
}

std::uniform_real_distribution<> unf(0.0, 1.0);
float uniform(){
  return unf(gen);
}

float istrue(float prob){
  return (uniform() < prob);
}

// Vectorized Functions

glm::vec2 vec2(){
  return glm::vec2(uniform(), uniform());
}

glm::vec3 vec3(){
  return glm::vec3(uniform(), uniform(), uniform());
}

};  //end of namespace dist
};  //end of namespace soil

#endif
