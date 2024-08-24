#ifndef SOILLIB_PARTICLE_WATER
#define SOILLIB_PARTICLE_WATER

#include <soillib/soillib.hpp>
#include <soillib/particle/particle.hpp>
#include <soillib/particle/cascade.hpp>

#include <soillib/core/node.hpp>

#include <soillib/node/cached.hpp>
#include <soillib/node/constant.hpp>
#include <soillib/node/computed.hpp>

#include <soillib/node/algorithm/normal.hpp>

#include <soillib/core/buffer.hpp>
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

  soil::index index;
  soil::node height;     //!< Height Array
  soil::node momentum;   //!< Momentum Array
  soil::node momentum_track;
  soil::node discharge;  //!< Discharge Array
  soil::node discharge_track;
  soil::node resistance; //!< Resistance Value
  soil::node maxdiff;    //!< Maximum Settling Height Difference
  soil::node settling;   //!< Settling Rate

  void add(const size_t index, const float value, const matrix_t matrix){
    soil::typeselect(height.type(), [self=this, index, value]<typename S>(){
      auto height = std::get<soil::cached>(self->height._node).as<float>();
      height.buffer[index] += value;
    });
  }

};

// WaterParticle Properties

struct WaterParticle_c {

  size_t maxAge = 1024;
  float evapRate = 0.001;
  float depositionRate = 0.05;
  float minVol = 0.001;
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

  void track(model_t& model);
};

bool WaterParticle::move(model_t& model, const WaterParticle_c& param){

  // Termination Checks

  const glm::ivec2 ipos = pos;
  const size_t index = model.index.flatten<2>(ipos);

  if(model.index.oob<2>(ipos))
    return false;


  if(age > param.maxAge){

    /*
    model.add(index, sediment, matrix);

    cascade_model_t casc{
      model.shape,
      model.height,
      model.maxdiff,
      model.settling
    };
    soil::cascade(casc, ipos);
    */

    return false;
  }

  if(volume < param.minVol){
    model.add(index, sediment, matrix);
    
    cascade_model_t casc{
      model.index,
      model.height,
      model.maxdiff,
      model.settling
    };
    soil::cascade(casc, ipos);

    return false;
  }

  // Apply Forces to Particle

  static auto normal = soil::normal(model.index, model.height);
  const glm::vec3 n = normal(ipos);

  const glm::vec2 fspeed = model.momentum.template operator()<vec2>(index);
  const float discharge = erf(0.4f * model.discharge.template operator()<float>(index));

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

void WaterParticle::track(model_t& model){

  if(model.index.oob<2>(this->pos))
    return;

  const size_t index = model.index.flatten<2>(this->pos);

  {
    auto cached = std::get<soil::cached>(model.discharge_track._node);
    soil::buffer_t<float> buffer = cached.as<float>().buffer;
    buffer[index] += this->volume;
  }

  {
    auto cached = std::get<soil::cached>(model.momentum_track._node);
    soil::buffer_t<vec2> buffer = cached.as<vec2>().buffer;
    buffer[index] += this->volume * this->speed;
  }

}

bool WaterParticle::interact(model_t& model, const WaterParticle_c& param){

  // Termination Checks

  const glm::ivec2 ipos = opos;
  const size_t index = model.index.flatten<2>(ipos);

  if(model.index.oob<2>(ipos))
    return false;

  const float discharge = erf(0.4f * model.discharge.template operator()<float>(index));
  const float resistance = model.resistance.template operator()<float>(index);

  //Out-Of-Bounds

  float h2;
  if(model.index.oob<2>(pos))
    h2 = 0.99f*model.height.template operator()<float>(index);
  else {
    const size_t index = model.index.flatten<2>(pos);
    h2 = model.height.template operator()<float>(index);
  }

  //Mass-Transfer (in MASS)
  float c_eq = (1.0f+param.entrainment*discharge)*(model.height.template operator()<float>(index)-h2);
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
    model.index,
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
