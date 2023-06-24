#ifndef SOILLIB_PARTICLE
#define SOILLIB_PARTICLE

namespace soil {

// Particle Base-Class

struct Particle {

  size_t age = 0;
  bool isAlive = true;

  template<typename T> bool move(T& map);
  template<typename T> bool interact(T& map);

};

};  //  end of namespace soil

#endif
