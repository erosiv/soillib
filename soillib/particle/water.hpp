#ifndef SOILLIB_PARTICLE_WATER
#define SOILLIB_PARTICLE_WATER

#include <soillib/particle/particle.hpp>
#include <soillib/model/physics/cascade.hpp>

namespace soil {

// WaterParticle Properties

struct WaterParticle_c {

  size_t maxAge = 1024;
  float evapRate = 0.001;
  float depositionRate = 0.1;
  float minVol = 0.01;
  float entrainment = 10.0f;
  float gravity = 2.0f;
  float momentumTransfer = 1.0f;

} static water_c;

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

  template<typename T>
  bool move(T& world, WaterParticle_c& param);

  template<typename T>
  bool interact(T& world, WaterParticle_c& param);

};

template<typename T>
bool WaterParticle::move(T& world, WaterParticle_c& param){

  const glm::ivec2 ipos = pos;
  cell* cell = world.map.get(ipos);
  if(cell == NULL){
    return false;
  }

  const glm::vec3 n = world.normal(ipos);

  // Termination Checks

  if(age > param.maxAge){
    cell->height += sediment;
    return false;
  }

  if(volume < param.minVol){
    cell->height += sediment;
    return false;
  }

  // Apply Forces to Particle

  // Gravity Force

  speed += param.gravity*glm::vec2(n.x, n.z)/volume;

  // Momentum Transfer Force

  glm::vec2 fspeed = glm::vec2(cell->momentumx, cell->momentumy);
  if(length(fspeed) > 0 && length(speed) > 0)
    speed += param.momentumTransfer*dot(normalize(fspeed), normalize(speed))/(volume + cell->discharge)*fspeed;

  // Dynamic Time-Step, Update

  if(length(speed) > 0)
    speed = (sqrt(2.0f))*normalize(speed);

  opos = pos;
  pos  += speed;

  // Update Discharge, Momentum Tracking Maps

  cell->discharge_track += volume;
  cell->momentumx_track += volume*speed.x;
  cell->momentumy_track += volume*speed.y;

  return true;

}

template<typename T>
bool WaterParticle::interact(T& world, WaterParticle_c& param){

  const glm::ivec2 ipos = opos;
  cell* cell = world.map.get(ipos);
  if(cell == NULL)
    return false;

  //Out-Of-Bounds
  float h2;
  if(world.map.oob(pos))
    h2 = cell->height-0.002;
  else
    h2 = world.height(pos);

  //Mass-Transfer (in MASS)
  float c_eq = (1.0f+param.entrainment*world.discharge(ipos))*(cell->height-h2);
  if(c_eq < 0) c_eq = 0;
  float cdiff = (c_eq - sediment);

  // Effective Parameter Set

  float effD = param.depositionRate*(1.0f - cell->rootdensity);
  if(effD < 0) effD = 0;


  sediment += effD*cdiff;
  cell->height -= effD*cdiff;

  //Evaporate (Mass Conservative)
  sediment /= (1.0-param.evapRate);
  volume *= (1.0-param.evapRate);

  //Out-Of-Bounds
  if(world.map.oob(pos)){
    volume = 0.0;
    return false;
  }

  soil::phys::cascade(world.map, pos);

  age++;
  return true;

}

} // end of namespace

#endif