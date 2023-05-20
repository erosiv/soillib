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


/*

static bool initialized = false;

template<typename T>
void init(T& map, int SEED){

  // Initialize Noise-Source


  // Zero-Out

  for(auto [cell, pos]: map.slice){
    cell.height = 0.0f;
  }

  // Add Layers of Noise

  float frequency = 1.0f;
  float scale = 0.6f;

  for(size_t o = 0; o < 8; o++){

    source.SetFrequency(frequency);

    for(auto [cell, pos]: map.slice){

      vec2 p = vec2(pos)/vec2(map.dimension);
      cell.height += scale*source.GetNoise(p.x, p.y, (float)(SEED%10000));

    }

    frequency *= 2;
    scale *= 0.6;

  }

  // Normalize

  float min = 0.0f;
  float max = 0.0f;

  for(auto [cell, pos]: map.slice){
    min = (min < cell.height)?min:cell.height;
    max = (max > cell.height)?max:cell.height;
  }

  for(auto [cell, pos]: map.slice){
    cell.height = ((cell.height - min)/(max - min));
  }

  
  // Add Gaussian

  for(auto [cell, pos]: node.s){
    vec2 p = vec2(node.pos+lodsize*pos)/vec2(tileres);
    vec2 c = vec2(node.pos+tileres/ivec2(4, 2))/vec2(tileres);
    float d = length(p-c);
    cell.height = exp(-d*d*tilesize*0.2);
  }
  
}

*/

};  // end of namespace noise
};  // end of namespace soil

#endif
