#ifndef SOILLIB_LAYER_COMPUTED_NOISE
#define SOILLIB_LAYER_COMPUTED_NOISE

#include <soillib/core/buffer.hpp>
#include <soillib/core/index.hpp>
#include <soillib/core/types.hpp>

#include <soillib/core/node.hpp>

#pragma GCC diagnostic ignored "-Waggressive-loop-optimizations"
#include <soillib/external/FastNoiseLite.h>

namespace soil {

namespace {

//! \todo make this configurable and exposed
struct sampler_t {

  FastNoiseLite::NoiseType ntype = FastNoiseLite::NoiseType_OpenSimplex2;
  FastNoiseLite::FractalType ftype = FastNoiseLite::FractalType_FBm;

  float frequency = 1.0f;  // Frequency
  int octaves = 8;         // Set of Scales
  float gain = 0.6f;       // Scale Multiplier
  float lacunarity = 2.0f; // Frequency Multiplier

  float min = -2.0f;  // Minimum Value
  float max = 2.0f;   // Maximum Value
  float bias = 0.0f;  // Add-To-Value
  float scale = 1.0f; // Multiply-By-Value
};

} // namespace

struct noise {

  noise() {
    source.SetNoiseType(cfg.ntype);
    source.SetFractalType(cfg.ftype);
    source.SetFrequency(cfg.frequency);
    source.SetFractalOctaves(cfg.octaves);
    source.SetFractalGain(cfg.gain);
    source.SetFractalLacunarity(cfg.lacunarity);
  }

  noise(const float seed): seed{seed} {
    source.SetNoiseType(cfg.ntype);
    source.SetFractalType(cfg.ftype);
    source.SetFrequency(cfg.frequency);
    source.SetFractalOctaves(cfg.octaves);
    source.SetFractalGain(cfg.gain);
    source.SetFractalLacunarity(cfg.lacunarity);
  }

  //! Single Sample Value
  float operator()(const soil::ivec2 pos) {

    const auto ext = soil::vec2{512, 512};
    const auto &cfg = this->cfg;

    float val = this->source.GetNoise(pos[0] / (float)ext[0], pos[1] / (float)ext[1], this->seed);

    // Clamp Value
    val = cfg.bias + cfg.scale * val;
    if (val < cfg.min)
      val = cfg.min;
    if (val > cfg.max)
      val = cfg.max;
    return val;
  }

private:
  float seed;
  FastNoiseLite source;
  sampler_t cfg;
};

// Noise Node Factory Function

soil::node make_noise(const soil::index index, const float seed) {

  return select(index.type(), [index, seed]<typename T>() -> soil::node {
    if constexpr (std::same_as<typename T::vec_t, soil::ivec2>) {

      using func_t = soil::map_t<float>::func_t;
      using param_t = soil::map_t<float>::param_t;

      const func_t func = [index, seed](const param_t& in, const size_t i) -> float {

        soil::noise noise(seed);
        auto index_t = index.as<T>();
        soil::ivec2 position = index_t.unflatten(i);
        return noise.operator()(position);

      };

      soil::map map = soil::map(func);
      return soil::node(map, {});

    } else
      throw std::invalid_argument("can't extract a full noise buffer from a non-2D index");

  });

}

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