#ifndef SOILLIB_PARTICLE_WATER
#define SOILLIB_PARTICLE_WATER

#include <soillib/soillib.hpp>
#include <soillib/particle/particle.hpp>
#include <soillib/particle/cascade.hpp>
// #include <soillib/model/cascade.hpp>

#include <soillib/layer/layer.hpp>
#include <soillib/layer/constant.hpp>
#include <soillib/layer/normal.hpp>

#include <soillib/util/array.hpp>
#include <soillib/matrix/matrix.hpp>

namespace soil {

/*

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
*/

//! water_particle_t is a type that contains the references
//! to all the layers that are required for executing hydraulic erosion.
//! This is effectively similar to the previous map struct, except that
//! the layers themselves provide the interface for sampling.
//!
struct water_particle_t {
  
  using matrix_t = soil::matrix::singular;

  soil::shape shape;
  soil::array height;         //!< Height Array
  soil::array momentum;       //!< Momentum Array
  soil::array discharge;      //!< Discharge Array
  soil::constant resistance;  //!< Resistance Value
  soil::constant maxdiff;
  soil::constant settling;

  void add(const size_t index, const float value, const matrix_t matrix){
    auto _height = std::get<soil::array_t<float>>(this->height._array);
    _height[index] += value;// / 80.0f;
  }

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

  //! \todo replace this with a variant (of course)
  using model_t = soil::water_particle_t;
  using matrix_t = soil::matrix::singular;

  WaterParticle(glm::vec2 pos)
    :pos(pos){}   // Construct at Position
 
  // Properties

  glm::vec2 pos;
  glm::vec2 opos;
  glm::vec2 speed = glm::vec2(0.0);

  float volume = 1.0;                   // Droplet Water Volume
  float sediment = 0.0;                 // Droplet Sediment Concentration
  matrix_t matrix;

  // Main Methods

  //template<typename T>
  bool move(model_t& model, const WaterParticle_c& param);

  //template<typename T>
  bool interact(model_t& model, const WaterParticle_c& param);

};

bool WaterParticle::move(model_t& model, const WaterParticle_c& param){

  // Termination Checks

  const glm::ivec2 ipos = pos;
  auto apos = std::array<int, 2>{ipos.x, ipos.y};
  const size_t index = model.shape.flat<2>(apos);

  if(model.shape.oob(ipos))
    return false;

  if(age > param.maxAge){
    model.add(index, sediment, matrix);

    cascade_model_t casc{
      model.shape,
      model.height,
      model.maxdiff,
      model.settling
    };
    soil::cascade(casc, ipos);

    return false;
  }

  if(volume < param.minVol){
    model.add(index, sediment, matrix);
    
    cascade_model_t casc{
      model.shape,
      model.height,
      model.maxdiff,
      model.settling
    };
    soil::cascade(casc, ipos);

    return false;
  }

  // Apply Forces to Particle

  const glm::vec3 n = soil::normal::sub()(model.height, ipos);

  const glm::vec2 fspeed = std::get<vec2>(model.momentum[index]);
  const float discharge = erf(0.4f * std::get<float>(model.discharge[index]));

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

bool WaterParticle::interact(model_t& model, const WaterParticle_c& param){

  // Termination Checks

  const glm::ivec2 ipos = opos;
  const size_t index = model.shape.flat(ipos);

  if(model.shape.oob(ipos))
    return false;

  const float discharge = erf(0.4f * std::get<float>(model.discharge[index]));
  const float resistance = std::get<float>(model.resistance(index));

  //Out-Of-Bounds

  float h2;
  if(model.shape.oob(pos))
    h2 = 0.99f*std::get<float>(model.height[index]);
  else {
    const size_t index = model.shape.flat(pos);
    h2 = std::get<float>(model.height[index]);
  }

  //Mass-Transfer (in MASS)
  float c_eq = (1.0f+param.entrainment*discharge)*(std::get<float>(model.height[index])-h2);
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

    if(effD*cdiff < -sediment) // Only Use Available
      cdiff = -sediment/effD;

  } else if(effD*cdiff > 0){

    auto wmatrix = matrix_t{};
    // wmatrix = world.matrix(ipos);

    matrix = (matrix*sediment + wmatrix*(effD*cdiff))/(sediment + effD*cdiff);

  }

  // Add Sediment Mass to Map, Particle

  sediment += effD*cdiff;
  model.add(index, -effD*cdiff, matrix);

  //Evaporate (Mass Conservative)

  volume *= (1.0-param.evapRate);

  // New Position Out-Of-Bounds
  cascade_model_t casc{
    model.shape,
    model.height,
    model.maxdiff,
    model.settling
  };
  soil::cascade(casc, ipos);

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
