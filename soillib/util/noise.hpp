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
  }

  inline float get(glm::vec3 pos) {
    return source.GetNoise(pos.x, pos.y, pos.z);
  }

};

};  // end of namespace noise
};  // end of namespace soil

#endif
