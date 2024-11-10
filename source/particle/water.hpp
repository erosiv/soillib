#ifndef SOILLIB_PARTICLE_WATER
#define SOILLIB_PARTICLE_WATER

#include <soillib/particle/cascade.hpp>
#include <soillib/particle/particle.hpp>
#include <soillib/soillib.hpp>

#include <soillib/core/model.hpp>
#include <soillib/core/node.hpp>

#include <soillib/node/normal.hpp>

#include <soillib/core/buffer.hpp>
#include <soillib/core/matrix.hpp>

namespace soil {

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

  using matrix_t = soil::matrix::singular;

  WaterParticle(glm::vec2 pos)
      : pos(pos) {} // Construct at Position

  // Properties

  glm::vec2 pos;
  glm::vec2 opos;
  glm::vec2 speed = glm::vec2(0.0);

  float volume = 1.0;   // Droplet Water Volume
  float sediment = 0.0; // Droplet Sediment Concentration
  matrix_t matrix;

  // Main Methods

  // template<typename T>
  bool move(soil::model &model, const WaterParticle_c &param);

  // template<typename T>
  bool interact(soil::model &model, const WaterParticle_c &param);

  void track(soil::model &model);
};

bool WaterParticle::move(soil::model &model, const WaterParticle_c &param) {

  // Termination Checks

  const glm::ivec2 ipos = pos;
  if (model.index.oob<2>(ipos))
    return false;

  const size_t index = model.index.flatten<2>(ipos);

  if (age > param.maxAge) {

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

  if (volume < param.minVol) {
    model.add(index, sediment, matrix);
    soil::cascade(model, ipos);
    return false;
  }

  // Apply Forces to Particle

  static auto normal = soil::normal::make_node(model.index, model[soil::HEIGHT]);
  const glm::vec3 n = normal.val<glm::vec3>(model.index.flatten<2>(ipos));

  const glm::vec2 fspeed = model[soil::MOMENTUM].val<vec2>(index);
  const float discharge = erf(0.4f * model[soil::DISCHARGE].val<float>(index));

  // Gravity Force

  speed += param.gravity * glm::vec2(n.x, n.y) / volume;

  // Momentum Transfer Force

  if (length(fspeed) > 0 && length(speed) > 0)
    speed += param.momentumTransfer * dot(normalize(fspeed), normalize(speed)) / (volume + discharge) * fspeed;

  // Dynamic Time-Step, Update

  if (length(speed) > 0)
    speed = (sqrt(2.0f)) * normalize(speed);

  opos = pos;
  pos += speed;

  return true;
}

void WaterParticle::track(soil::model &model) {

  if (model.index.oob<2>(this->pos))
    return;

  const size_t index = model.index.flatten<2>(this->pos);

  {
    model[soil::DISCHARGE_TRACK].ref<float>(index) += this->volume;
  }

  {
    model[soil::MOMENTUM_TRACK].ref<vec2>(index) += this->volume * this->speed;
  }
}

bool WaterParticle::interact(soil::model &model, const WaterParticle_c &param) {

  // Termination Checks

  const glm::ivec2 ipos = opos;
  const size_t index = model.index.flatten<2>(ipos);

  if (model.index.oob<2>(ipos))
    return false;

  const float discharge = erf(0.4f * model[soil::DISCHARGE].val<float>(index));
  const float resistance = model[soil::RESISTANCE].val<float>(index);

  // Out-Of-Bounds

  float h2;
  if (model.index.oob<2>(pos))
    h2 = 0.99f * model[soil::HEIGHT].val<float>(index);
  else {
    const size_t index = model.index.flatten<2>(pos);
    h2 = model[soil::HEIGHT].val<float>(index);
  }

  // Mass-Transfer (in MASS)
  float c_eq = (1.0f + param.entrainment * discharge) * (model[soil::HEIGHT].val<float>(index) - h2);
  if (c_eq < 0)
    c_eq = 0;

  float cdiff = (c_eq * volume - sediment);

  // Effective Parameter Set

  float effD = param.depositionRate * (1.0f - resistance);
  if (effD < 0)
    effD = 0;

  // Compute Actual Mass Transfer

  // Add Sediment to Map

  if (effD * cdiff < 0) {

    if (effD * cdiff < -sediment) // Only Use Available
      cdiff = -sediment / effD;

  } else if (effD * cdiff > 0) {

    auto wmatrix = matrix_t{};
    // wmatrix = world.matrix(ipos);

    matrix = (matrix * sediment + wmatrix * (effD * cdiff)) / (sediment + effD * cdiff);
  }

  // Add Sediment Mass to Map, Particle

  sediment += effD * cdiff;
  model.add(index, -effD * cdiff, matrix);

  // Evaporate (Mass Conservative)

  volume *= (1.0 - param.evapRate);

  // New Position Out-Of-Bounds
  soil::cascade(model, ipos);

  age++;
  return true;
}

// Configuration Loading

#ifdef SOILLIB_IO_YAML

template<>
struct soil::io::yaml::cast<WaterParticle_c> {
  static WaterParticle_c As(soil::io::yaml &node) {
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

} // namespace soil

#endif
