#ifndef SOILLIB_PARTICLE_WATER
#define SOILLIB_PARTICLE_WATER

#include <soillib/soillib.hpp>
#include <soillib/particle/particle.hpp>
#include <soillib/model/cascade.hpp>

namespace soil {

// Hydrologically Erodable Map Constraints

template<typename T>
concept WaterParticle_t = requires(T t){
  // Measurement Methods
  { t.oob(glm::ivec2()) } -> std::same_as<bool>;
  { t.height(glm::ivec2()) } -> std::same_as<float>;
  { t.normal(glm::ivec2()) } -> std::convertible_to<glm::vec3>;
  // Specific Physics Requirements
  { t.discharge(glm::ivec2()) } -> std::same_as<float>;
  { t.momentum(glm::ivec2()) } -> std::convertible_to<glm::vec2>;
  { t.resistance(glm::ivec2()) } -> std::same_as<float>;
  // Add Method
  { t.add(glm::ivec2(), float()) } -> std::same_as<void>;
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

struct WaterParticle: soil::Particle {

  WaterParticle(glm::vec2 _pos){ pos = _pos; }   // Construct at Position

  // Properties

  glm::vec2 pos;
  glm::vec2 opos;
  glm::vec2 speed = glm::vec2(0.0);

  float volume = 1.0;                   // Droplet Water Volume
  float sediment = 0.0;                 // Droplet Sediment Concentration

  // Main Methods

  template<WaterParticle_t T>
  bool move(T& world, WaterParticle_c& param);

  template<typename T>
  bool interact(T& world, WaterParticle_c& param);

};

template<WaterParticle_t T>
bool WaterParticle::move(T& world, WaterParticle_c& param){

  // Termination Checks

  const glm::ivec2 ipos = pos;
  if(world.oob(ipos))
    return false;

  if(age > param.maxAge){
    world.add(ipos, sediment);
    return false;
  }

  if(volume < param.minVol){
    world.add(ipos, sediment);
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
    speed += param.momentumTransfer*dot(normalize(fspeed), normalize(speed))/(volume + discharge)*fspeed;

  // Dynamic Time-Step, Update

  if(length(speed) > 0)
    speed = (sqrt(2.0f))*normalize(speed);

  opos = pos;
  pos  += speed;

  return true;

}

template<typename T>
bool WaterParticle::interact(T& world, WaterParticle_c& param){

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

  float cdiff = (c_eq - sediment);

  // Effective Parameter Set

  float effD = param.depositionRate*(1.0f - resistance);
  if(effD < 0)
    effD = 0;

  sediment += effD*cdiff;
  world.add(ipos, -effD*cdiff);

  //Evaporate (Mass Conservative)
  sediment /= (1.0-param.evapRate);
  volume *= (1.0-param.evapRate);

  //Out-Of-Bounds
  if(world.oob(pos)){
    volume = 0.0;
    return false;
  }

  soil::phys::cascade(world, pos);

  age++;
  return true;

}

// Configuration Loading

#ifdef SOILLIB_IO_YAML

bool operator<<(WaterParticle_c& conf, soil::io::yaml::node& node){
  try {
    conf.maxAge = node["max-age"].As<int>();
    conf.evapRate = node["evap-rate"].As<float>();
    conf.depositionRate = node["deposition-rate"].As<float>();
    conf.minVol = node["min-vol"].As<float>();
    conf.entrainment = node["entrainment"].As<float>();
    conf.gravity = node["gravity"].As<float>();
    conf.momentumTransfer = node["momentum-transfer"].As<float>();
  } catch(soil::io::yaml::exception& e){
    return false;
  }
  return true;
}

#endif

} // end of namespace

#endif
