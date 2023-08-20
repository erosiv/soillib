#ifndef SOILLIB_PARTICLE_WATER
#define SOILLIB_PARTICLE_WATER

#include <soillib/soillib.hpp>
#include <soillib/particle/particle.hpp>

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

  if(age > param.maxAge)
    return false;

  if(volume == 0)
    return false;
  
  // Apply Forces to Particle

  const glm::vec3 n = world.normal(ipos);
  const glm::vec2 fspeed = world.momentum(ipos);
  const float discharge = world.discharge(ipos);

  // Gravity Force

  speed += world.gravity(ipos)*glm::vec2(n.x, n.z)/volume;

  // Momentum Transfer Force

  if(length(fspeed) > 0 && length(speed) > 0)
    speed += world.transfer(ipos)*dot(normalize(fspeed), normalize(speed))/(volume + discharge)*fspeed;

  // Dynamic Time-Step, Update

  if(length(speed) > 0)
    speed = normalize(speed);

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
  matrix = world.matrix(ipos);

  //Out-Of-Bounds

  // Sediment Transport

  if(!matrix.is_water){

    float h2;
    if(world.oob(pos))
      h2 = 0.99*world.height(ipos);
    else
      h2 = world.height(pos);


    // Non-Negative Values

    const float c_eq = glm::max(0.0f, (1.0f+param.entrainment*discharge)*(world.height(ipos)-h2));
    const float effD = glm::max(0.0f, param.depositionRate*(1.0f - resistance));
    
    // Capped at Sediment

    const float transfer = glm::min(sediment, effD*(sediment - c_eq*volume));

    // Transfer

    sediment -= transfer;
    world.add(ipos, transfer, matrix);

  }

  // Water Transport

  if(matrix.is_water){

    float h2;
    if(world.oob(pos))
      h2 = 0.99*world.height(ipos);
    else{
      h2 = world.height(pos);     
    }

    // Non-Negative Values

    const float effD = 0.5;
    const float c_eq = glm::max(0.0f, volume + effD*(world.height(ipos)-h2)/world.config.waterscale);

    // Capped at Volume

    float transfer = (volume - c_eq);

    if(transfer > 0){
      transfer = glm::min(volume, transfer);
    }

    if(transfer < 0){
      transfer = -world.maxremove(ipos, -transfer*world.config.waterscale)/world.config.waterscale;
    }

    // Transfer

    world.add(ipos, transfer*world.config.waterscale, matrix);
    volume -= transfer;

  }

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
