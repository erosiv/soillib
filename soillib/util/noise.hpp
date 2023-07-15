#ifndef SOILLIB_UTIL_NOISE
#define SOILLIB_UTIL_NOISE

#include <soillib/external/FastNoiseLite.h>

/*

  Technique for generating noise based maps
  you want to be able to sample a specific
  composition.

  For now, we will hard-code these functions
  but in the future we will parameterize them.

*/

namespace soil {
namespace noise {

struct sampler_t {

  FastNoiseLite::NoiseType ntype = FastNoiseLite::NoiseType_OpenSimplex2;
  FastNoiseLite::FractalType ftype = FastNoiseLite::FractalType_FBm;

  float frequency = 1.0f;   // Frequency
  int octaves = 3;          // Set of Scales
  float gain = 0.6f;        // Scale Multiplier
  float lacunarity = 2.0f;  // Frequency Multiplier

  float min =  0.0f;        //Minimum Value
  float max = 1.0f;         //Maximum Value
  float bias = 0.0f;        //Add-To-Value
  float scale = 1.0f;       //Multiply-By-Value

};

struct sampler {

  FastNoiseLite source;
  sampler_t cfg;

  sampler(){
    source.SetNoiseType(cfg.ntype);
    source.SetFractalType(cfg.ftype);
    source.SetFrequency(cfg.frequency);
    source.SetFractalOctaves(cfg.octaves);
    source.SetFractalGain(cfg.gain);
    source.SetFractalLacunarity(cfg.lacunarity);
  }

  sampler(sampler_t cfg):sampler(){
    this->cfg = cfg;
    source.SetNoiseType(cfg.ntype);
    source.SetFractalType(cfg.ftype);
    source.SetFrequency(cfg.frequency);
    source.SetFractalOctaves(cfg.octaves);
    source.SetFractalGain(cfg.gain);
    source.SetFractalLacunarity(cfg.lacunarity);
  }

  inline float get(glm::vec3 pos) {
    float val = cfg.bias + cfg.scale * source.GetNoise(pos.x, pos.y, pos.z);
    if(val < cfg.min) val = cfg.min;
    if(val > cfg.max) val = cfg.max;
    return val;
  }

};

};  // end of namespace noise
};  // end of namespace soil

// Configuration Loading

#ifdef SOILLIB_IO_YAML

bool operator<<(soil::noise::sampler_t& conf, soil::io::yaml::node& node){
  try {
    conf.octaves << node["octaves"];
    conf.frequency << node["frequency"];
    conf.gain << node["gain"];
    conf.lacunarity << node["lacunarity"];
    conf.min << node["min"];
    conf.max << node["max"];
    conf.bias << node["bias"];
    conf.scale << node["scale"];
  } catch(soil::io::yaml::exception& e){
    return false;
  }
  return true;
}

#endif

#endif
