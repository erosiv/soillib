#ifndef SOILLIB_PARTICLE_WIND
#define SOILLIB_PARTICLE_WIND

#include <soillib/model/cascade.hpp>
#include <soillib/particle/particle.hpp>
#include <soillib/soillib.hpp>

namespace soil {

// WindParticle Properties

struct WindParticle_c {

  size_t maxAge = 512;
  float boundaryLayer = 2.0f;
  float suspension = 0.05f;
  float gravity = 0.1;

} static wind_c;

// WindParticle Definition

template<typename M>
struct WindParticle: soil::Particle {

  WindParticle(glm::vec2 _pos) { pos = glm::vec3(_pos.x, 0.0f, _pos.y); }

  // Properties

  glm::vec3 pos;
  glm::vec3 opos;
  glm::vec3 speed = glm::vec3(0);
  glm::vec3 pspeed = glm::normalize(glm::vec3(1, 0, 0));

  int age = 0;
  float sediment = 0.0; // Sediment Mass
  M matrix;

  // Main Methods

  template<typename T>
  bool move(T &world, WindParticle_c &param);

  template<typename T>
  bool interact(T &world, WindParticle_c &param);
};

template<typename M>
template<typename T>
bool WindParticle<M>::move(T &world, WindParticle_c &param) {

  // Termination Checks

  const glm::ivec2 ipos = glm::ivec2(pos.x, pos.z);
  if (world.oob(ipos))
    return false;

  if (age > param.maxAge) {
    world.add(ipos, sediment, matrix);
    soil::phys::cascade<M>(world, ipos);
    return false;
  }

  // Compute Movement

  const float height = world.height(ipos);
  if (age == 0 || pos.y < height)
    pos.y = height;

  const glm::vec3 n = world.normal(ipos);
  const float hfac = exp(-(pos.y - height) / param.boundaryLayer);
  const float shadow = 1.0f - glm::max(0.0f, dot(normalize(pspeed), n));
  const float collision = glm::max(0.0f, -dot(normalize(speed), n));
  const glm::vec3 rspeed = cross(n, cross((1.0f - collision) * speed, n));

  // Apply Base Prevailign Wind-Speed w. Shadowing

  speed += 0.05f * ((0.1f + 0.9f * shadow) * pspeed - speed);

  // Apply Gravity

  if (pos.y > height)
    speed.y -= param.gravity * sediment;

  // Compute Collision Factor

  // Compute Redirect Velocity

  // Speed is accelerated by terrain features

  speed += 0.9f * (shadow * mix(pspeed, rspeed, shadow * hfac) - speed);

  // Turbulence

  speed += 0.1f * hfac * collision * (glm::vec3(rand() % 1001, rand() % 1001, rand() % 1001) - 500.0f) / 500.0f;

  // Speed is damped by drag

  speed *= (1.0f - 0.3 * sediment);

  // Move

  opos = pos;
  pos += speed;

  return true;
};

template<typename M>
template<typename T>
bool WindParticle<M>::interact(T &world, WindParticle_c &param) {

  // Termination Checks

  const glm::ivec2 cpos = glm::ivec2(pos.x, pos.z);
  const glm::ivec2 ipos = glm::ivec2(opos.x, opos.z);

  if (world.oob(cpos))
    return false;

  // Compute Mass Transport

  const glm::vec3 n = world.normal(cpos);
  const float height = world.height(cpos);

  const float hfac = exp(-(pos.y - height) / param.boundaryLayer);
  const float collision = glm::max(0.0f, -dot(normalize(speed), n));
  const float force = glm::max(0.0f, -dot(normalize(speed), n) * length(speed));

  float lift = (1.0f - collision) * length(speed);

  float capacity = 10 * (force * hfac + 0.02f * lift * hfac);

  // Mass Transfer to Equilibrium

  float diff = capacity - sediment;

  sediment += param.suspension * diff;
  world.add(cpos, -param.suspension * diff, matrix);

  //  World::cascade(ipos);

  // soil::phys::cascade_c::maxdiff = 0.4;
  // soil::phys::cascade_c::settling = 0.1;
  soil::phys::cascade<M>(world, cpos);

  age++;
  return true;
}

// Configuration Loading

#ifdef SOILLIB_IO_YAML

template<>
struct soil::io::yaml::cast<WindParticle_c> {
  static WindParticle_c As(soil::io::yaml &node) {
    WindParticle_c config;
    config.maxAge = node["max-age"].As<int>();
    config.boundaryLayer = node["boundary-layer"].As<float>();
    config.suspension = node["suspension"].As<float>();
    config.gravity = node["gravity"].As<float>();
    return config;
  }
};

#endif

} // namespace soil

#endif
