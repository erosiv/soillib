#ifndef SOILLIB_PARTICLE_WIND
#define SOILLIB_PARTICLE_WIND

#include <soillib/soillib.hpp>
#include <soillib/particle/particle.hpp>
#include <soillib/model/cascade.hpp>

namespace soil {

// WindParticle Properties

struct WindParticle_c {

  size_t maxAge = 512;
  float boundaryLayer = 2.0f;
  float suspension = 0.05f;
  float gravity = 0.1;

} static wind_c;

// WindParticle Definition

struct WindParticle: soil::Particle {

  WindParticle(glm::vec2 _pos){ pos = glm::vec3(_pos.x, 0.0f, _pos.y); }

  // Properties

  glm::vec3 pos;
  glm::vec3 opos;
  glm::vec3 speed = glm::vec3(0);
  glm::vec3 pspeed = glm::normalize(glm::vec3(1, 0, 0));

  int age = 0;
  float sediment = 0.0;     //Sediment Mass

  // Main Methods

  template<typename T>
  bool move(T& world, WindParticle_c& param);

  template<typename T>
  bool interact(T& world, WindParticle_c& param);

};

template<typename T>
bool WindParticle::move(T& world, WindParticle_c& param){

  const glm::ivec2 ipos = glm::vec2(pos.x, pos.z);
  auto cell = world.map.get(ipos);
  if(cell == NULL){
    return false;
  }

  const glm::vec3 n = world.normal(ipos);

  // Termination Checks

  if(age++ > param.maxAge){
    return false;
  }

  if(age == 0 || pos.y < cell->height)
    pos.y = cell->height;

  // Compute Movement

  float hfac = exp(-(pos.y - cell->height)/param.boundaryLayer);
  if(hfac < 0)
    hfac = 0;


  // Apply Base Prevailign Wind-Speed w. Shadowing

  float shadow = dot(normalize(pspeed), n);
  if(shadow < 0)
    shadow = 0;
  shadow = 1.0f-shadow;

  speed += 0.05f*((0.1f+0.9f*shadow)*pspeed - speed);

  // Apply Gravity

  if(pos.y > cell->height)
    speed.y -= param.gravity*sediment;

  // Compute Collision Factor

  float collision = -dot(normalize(speed), n);
  if(collision < 0) collision = 0;

  // Compute Redirect Velocity

  glm::vec3 rspeed = cross(n, cross((1.0f-collision)*speed, n));

  // Speed is accelerated by terrain features

  speed += 0.9f*( shadow*mix(pspeed, rspeed, shadow*hfac) - speed);

  // Turbulence

  speed += 0.1f*hfac*collision*(glm::vec3(rand()%1001, rand()%1001, rand()%1001)-500.0f)/500.0f;

  // Speed is damped by drag

  speed *= (1.0f - 0.3*sediment);

  // Move

  opos = pos;
  pos += speed;

  // Update Momentum Tracking Maps

  cell->momentumx_track += speed.x;
  cell->momentumy_track += speed.y;
  cell->momentumz_track += speed.z;
  cell->massflow_track += sediment;

   // Compute Mass Transport

  float force = -dot(normalize(speed), n)*length(speed);
  if(force < 0)
    force = 0;

  float lift = (1.0f-collision)*length(speed);

  float capacity = force*hfac + 0.02f*lift*hfac;

  // Mass Transfer to Equilibrium

  float diff = capacity - sediment;
  cell->height -= param.suspension*diff;
  sediment += param.suspension*diff;

//  World::cascade(ipos);
  soil::phys::cascade_c::maxdiff = 0.002;
  soil::phys::cascade(world.map, ipos);
  soil::phys::cascade(world.map, ipos);

  return true;

};

template<typename T>
bool WindParticle::interact(T& map, WindParticle_c& param){

  return true;

}

// Configuration Loading

#ifdef SOILLIB_IO_YAML

template<>
struct soil::io::yaml::cast<WindParticle_c> {
  static WindParticle_c As(soil::io::yaml& node){
    WindParticle_c config;
    config.maxAge = node["max-age"].As<int>();
    config.boundaryLayer = node["boundary-layer"].As<float>();
    config.suspension = node["suspension"].As<float>();
    config.gravity = node["gravity"].As<float>();
    return config;
  }
};

#endif

} // end of namespace

#endif
