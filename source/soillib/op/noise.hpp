#ifndef SOILLIB_OP_NOISE
#define SOILLIB_OP_NOISE

#include <silt/core/shape.hpp>
#include <silt/core/tensor.hpp>
#include <silt/core/types.hpp>

#pragma GCC diagnostic ignored "-Waggressive-loop-optimizations"
#include <soillib/external/FastNoiseLite.h>

namespace soil {

//! \todo make this configurable and exposed
struct noise_param_t {

  void update() {
    source.SetNoiseType(ntype);
    source.SetFractalType(ftype);
    source.SetFrequency(frequency);
    source.SetFractalOctaves(octaves);
    source.SetFractalGain(gain);
    source.SetFractalLacunarity(lacunarity);
  }

  FastNoiseLite::NoiseType ntype = FastNoiseLite::NoiseType_OpenSimplex2;
  FastNoiseLite::FractalType ftype = FastNoiseLite::FractalType_FBm;

  FastNoiseLite source;
  float frequency = 1.0f;  // Frequency
  int octaves = 8;         // Set of Scales
  float gain = 0.6f;       // Scale Multiplier
  float lacunarity = 2.0f; // Frequency Multiplier
  float seed = 0.0f;
  silt::vec2 ext = {512, 512}; // Grid-Space Frequency

  //! Single Sample Value
  float operator()(const silt::ivec2 pos) {
    return this->source.GetNoise(pos[0] / ext[0], pos[1] / ext[1], this->seed);
  }
};

silt::tensor noise(const silt::shape shape, noise_param_t param) {

  if(shape.dim() != 2)
    throw std::invalid_argument("can't extract a full noise buffer from a non-2D index");
  
  silt::tensor_t<float> tensor_t(shape, silt::CPU);
  param.update();
  for (size_t i = 0; i < shape.elem(); ++i) {
    silt::ivec2 position = shape.unflatten(i);
    tensor_t[i] = param(position);
  }

  return silt::tensor(std::move(tensor_t));

}

}; // end of namespace soil

#endif