#ifndef SIMPLEHYDROLOGY_WATER
#define SIMPLEHYDROLOGY_WATER

#include <soillib/particle/particle.hpp>
#include <soillib/model/physics/cascade.hpp>

/*
SimpleHydrology - water.h

Defines our particle struct and
the method for descending / eroding
the landscape.
*/

struct Drop_c {

  size_t maxAge = 512;
  float evapRate = 0.001;
  float depositionRate = 0.1;
  float minVol = 0.01;
  float entrainment = 10.0f;
  float gravity = 2.0f;
  float momentumTransfer = 1.0f;

} drop_c;

struct Drop: soil::Particle {

  Drop(glm::vec2 _pos){ pos = _pos; }   // Construct at Position

  // Properties

  glm::vec2 pos;
  glm::vec2 opos;
  glm::vec2 speed = glm::vec2(0.0);

  float volume = 1.0;                   // Droplet Water Volume
  float sediment = 0.0;                 // Droplet Sediment Concentration

  // Main Methods

  template<typename T>
  bool move(T& map, Drop_c& param);

  template<typename T>
  bool interact(T& map, Drop_c& param);

};

template<typename T>
bool Drop::move(T& map, Drop_c& param){

  const glm::ivec2 ipos = pos;
  cell* cell = map.get(ipos);
  if(cell == NULL)
    return false;

  const glm::vec3 n = World::normal(ipos);

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
bool Drop::interact(T& map, Drop_c& param){

  const glm::ivec2 ipos = opos;
  cell* cell = map.get(ipos);
  if(cell == NULL)
    return false;

  //Out-Of-Bounds
  float h2;
  if(map.oob(pos))
    h2 = cell->height-0.002;
  else
    h2 = World::height(pos);

  //Mass-Transfer (in MASS)
  float c_eq = (1.0f+param.entrainment*World::discharge(ipos))*(cell->height-h2);
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
  if(World::map.oob(pos)){
    volume = 0.0;
    return false;
  }

  soil::phys::cascade(World::map, pos);

  age++;
  return true;

}

#endif
