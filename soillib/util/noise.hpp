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

static FastNoiseLite source;
static bool initialized = false;

template<typename T>
void init(T& map, int SEED){

  // Initialize Noise-Source

  source.SetNoiseType(FastNoiseLite::NoiseType_OpenSimplex2);
  source.SetFractalType(FastNoiseLite::FractalType_FBm);

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

  /*
  // Add Gaussian

  for(auto [cell, pos]: node.s){
    vec2 p = vec2(node.pos+lodsize*pos)/vec2(tileres);
    vec2 c = vec2(node.pos+tileres/ivec2(4, 2))/vec2(tileres);
    float d = length(p-c);
    cell.height = exp(-d*d*tilesize*0.2);
  }
  */
}

};  // end of namespace noise
};  // end of namespace soil

#endif
