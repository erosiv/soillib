#ifndef SOILLIB_OP_NOISE
#define SOILLIB_OP_NOISE

#include <soillib/core/buffer.hpp>
#include <soillib/core/index.hpp>
#include <soillib/core/types.hpp>

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
  soil::vec2 ext = {512, 512}; // Grid-Space Frequency

  //! Single Sample Value
  float operator()(const soil::ivec2 pos){
    return this->source.GetNoise(pos[0]/ext[0], pos[1]/ext[1], this->seed);
  }

};

struct noise {

  static soil::buffer make_buffer(const soil::index index, noise_param_t param) {

    return select(index.type(), [index, &param]<typename T>() -> soil::buffer {
      if constexpr (std::same_as<typename T::vec_t, soil::ivec2>) {

        auto index_t = index.as<T>();
        auto buffer_t = soil::buffer_t<float>(index_t.elem(), soil::CPU);

        param.update();
        for(size_t i = 0; i < index_t.elem(); ++i){
          soil::ivec2 position = index_t.unflatten(i);
          buffer_t[i] = param(position);
        }
        return soil::buffer(std::move(buffer_t));

      } else
        throw std::invalid_argument("can't extract a full noise buffer from a non-2D index");
    });

  }  

};

}; // end of namespace soil

// Configuration Loading

#ifdef SOILLIB_IO_YAML

template<>
struct soil::io::yaml::cast<soil::noise::sampler_t> {
  static soil::noise::sampler_t As(soil::io::yaml &node) {
    soil::noise::sampler_t sampler;
    sampler.frequency = node["frequency"].As<float>();
    sampler.octaves = node["octaves"].As<int>();
    sampler.lacunarity = node["lacunarity"].As<float>();
    sampler.min = node["min"].As<float>();
    sampler.max = node["max"].As<float>();
    sampler.bias = node["bias"].As<float>();
    sampler.scale = node["scale"].As<float>();
    return sampler;
  }
};

#endif
#endif