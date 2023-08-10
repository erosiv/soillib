#ifndef SOILLIB_PARTICLE_WATER
#define SOILLIB_PARTICLE_WATER

#include <soillib/soillib.hpp>
#include <soillib/particle/particle.hpp>
#include <soillib/model/cascade.hpp>

namespace soil {

// Hydrologically Erodable Map Constraints

template<typename T, typename M>
concept WaterParticle_t = requires(T t){
  { t.oob(glm::ivec2()) } -> std::same_as<bool>;
  { t.height(glm::ivec2()) } -> std::same_as<float>;
  { t.normal(glm::ivec2()) } -> std::convertible_to<glm::vec3>;
  { t.matrix(glm::ivec2()) } -> std::same_as<M>;
  { t.add(glm::ivec2(), float(), M()) } -> std::same_as<void>;
  { t.discharge(glm::ivec2()) } -> std::same_as<float>;
  { t.momentum(glm::ivec2()) } -> std::convertible_to<glm::vec2>;
  { t.resistance(glm::ivec2()) } -> std::same_as<float>;
};

// WaterParticle Properties

struct WaterParticle_c {

  size_t maxAge = 1024;
  float evapRate = 0.001;
  float depositionRate = 0.1;
  float minVol = 0.01;
  float entrainment = 10.0f;
  float gravity = 2.0f;
  float momentumTransfer = 1.0f;

};

// WaterParticle Definition

template<typename M>
struct WaterParticle: soil::Particle {

  WaterParticle(glm::vec2 pos)
    :pos(pos){}   // Construct at Position

  // Properties

  glm::vec2 pos;
  glm::vec2 opos;
  glm::vec2 speed = glm::vec2(0.0);

  float volume = 1.0;                   // Droplet Water Volume
  float sediment = 0.0;                 // Droplet Sediment Concentration
  M matrix;

  // Main Methods

  template<typename T>
  bool move(T& world, WaterParticle_c& param);

  template<typename T>
  bool interact(T& world, WaterParticle_c& param);

};

template<typename M>
template<typename T>
bool WaterParticle<M>::move(T& world, WaterParticle_c& param){

  // Termination Checks

  const glm::ivec2 ipos = pos;
  if(world.oob(ipos))
    return false;

  if(age > param.maxAge){
    world.add(ipos, sediment, matrix);
    soil::phys::cascade<M>(world, ipos);
    return false;
  }

  if(volume < param.minVol){
    world.add(ipos, sediment, matrix);
    soil::phys::cascade<M>(world, ipos);
    return false;
  }

  // Apply Forces to Particle

  const glm::vec3 n = world.normal(ipos);
  const glm::vec2 fspeed = world.momentum(ipos);
  const float discharge = world.discharge(ipos);

  // Gravity Force

  speed += param.gravity*glm::vec2(n.x, n.z)/volume;

  // Momentum Transfer Force

  if(length(fspeed) > 0 && length(speed) > 0)
    speed += world.transfer(ipos)*dot(normalize(fspeed), normalize(speed))/(volume + discharge)*fspeed;

  // Dynamic Time-Step, Update

  if(length(speed) > 0)
    speed = (sqrt(2.0f))*normalize(speed);

  opos = pos;
  pos  += speed;

  return true;

}

template<typename M>
template<typename T>
bool WaterParticle<M>::interact(T& world, WaterParticle_c& param){

  // Termination Checks

  const glm::ivec2 ipos = opos;
  if(world.oob(ipos))
    return false;

  const float discharge = world.discharge(ipos);
  const float resistance = world.resistance(ipos);

  //Out-Of-Bounds

  float h2;
  if(world.oob(pos))
    h2 = 0.99*world.height(ipos);
  else
    h2 = world.height(pos);

  //Mass-Transfer (in MASS)
  float c_eq = (1.0f+param.entrainment*discharge)*(world.height(ipos)-h2);
  if(c_eq < 0)
    c_eq = 0;

  float cdiff = (c_eq*volume - sediment);

  // Effective Parameter Set

  float effD = param.depositionRate*(1.0f - resistance);
  if(effD < 0)
    effD = 0;

  // Compute Actual Mass Transfer

  // Add Sediment to Map

  if(effD*cdiff < 0){

    if(matrix.is_water && effD*cdiff < -sediment) // Only Use Available
      cdiff = -sediment/effD;

  } else if(effD*cdiff > 0){

    matrix = world.matrix(ipos);//(matrix*sediment + world.matrix(ipos)*(effD*cdiff))/(sediment + effD*cdiff);

  }

  // Add Sediment Mass to Map, Particle

  sediment += effD*cdiff;
  world.add(ipos, -effD*cdiff, matrix);

  //Evaporate (Mass Conservative)

  volume *= (1.0-param.evapRate);

  // New Position Out-Of-Bounds

  soil::phys::cascade<M>(world, pos);

  age++;
  return true;

}

// Configuration Loading

#ifdef SOILLIB_IO_YAML

template<>
struct soil::io::yaml::cast<WaterParticle_c> {
  static WaterParticle_c As(soil::io::yaml& node){
    WaterParticle_c config;
    config.maxAge = node["max-age"].As<int>();
    config.evapRate = node["evap-rate"].As<float>();
    config.depositionRate = node["deposition-rate"].As<float>();
    config.minVol = node["min-vol"].As<float>();
    config.entrainment = node["entrainment"].As<float>();
    config.gravity = node["gravity"].As<float>();
    config.momentumTransfer = node["momentum-transfer"].As<float>();
    return config;
  }
};

#endif

} // end of namespace

#endif
