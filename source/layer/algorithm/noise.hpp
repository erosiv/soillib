#ifndef SOILLIB_LAYER_COMPUTED_NOISE
#define SOILLIB_LAYER_COMPUTED_NOISE

#include <soillib/util/types.hpp>
#include <soillib/util/shape.hpp>
#include <soillib/util/buffer.hpp>

#pragma GCC diagnostic ignored "-Waggressive-loop-optimizations"
#include <soillib/external/FastNoiseLite.h>

namespace soil {

namespace {

//! \todo make this configurable and exposed
struct sampler_t {

  FastNoiseLite::NoiseType ntype = FastNoiseLite::NoiseType_OpenSimplex2;
  FastNoiseLite::FractalType ftype = FastNoiseLite::FractalType_FBm;

  float frequency = 1.0f;   // Frequency
  int octaves = 8;          // Set of Scales
  float gain = 0.6f;        // Scale Multiplier
  float lacunarity = 2.0f;  // Frequency Multiplier

  float min =  -2.0f;        //Minimum Value
  float max = 2.0f;         //Maximum Value
  float bias = 0.0f;        //Add-To-Value
  float scale = 1.0f;       //Multiply-By-Value

};

}

struct noise {

  noise(){
    source.SetNoiseType(cfg.ntype);
    source.SetFractalType(cfg.ftype);
    source.SetFrequency(cfg.frequency);
    source.SetFractalOctaves(cfg.octaves);
    source.SetFractalGain(cfg.gain);
    source.SetFractalLacunarity(cfg.lacunarity);
  }

  noise(const soil::shape shape, const float seed):
  shape{shape},seed{seed}{
    source.SetNoiseType(cfg.ntype);
    source.SetFractalType(cfg.ftype);
    source.SetFrequency(cfg.frequency);
    source.SetFractalOctaves(cfg.octaves);
    source.SetFractalGain(cfg.gain);
    source.SetFractalLacunarity(cfg.lacunarity);
  }

  //noise(sampler_t cfg):noise(){
  //  this->cfg = cfg;
  //  source.SetNoiseType(cfg.ntype);
  //  source.SetFractalType(cfg.ftype);
  //  source.SetFrequency(cfg.frequency);
  //  source.SetFractalOctaves(cfg.octaves);
  //  source.SetFractalGain(cfg.gain);
  //  source.SetFractalLacunarity(cfg.lacunarity);
  //}

  //! Single Sample Value
  float operator()(const soil::ivec2 pos){
    return noise_impl(pos);
  }

  //! Bake a whole buffer!
  soil::buffer full(){
    buffer_t<float> out = buffer_t<float>{shape.elem()};
    auto _shape = std::get<soil::shape_t<2>>(shape._shape);
    for(const auto& pos: _shape.iter()){
      const size_t index = _shape.flat(pos);
      out[index] = this->operator()(glm::ivec2(pos[0], pos[1]));
    }
    return std::move(soil::buffer(std::move(out)));
  }

private:

  inline float noise_impl(const soil::vec2 pos) {
    float val;
    val = source.GetNoise(pos[0]/(float)shape[0], pos[1]/(float)shape[1], seed);
    val = cfg.bias + cfg.scale * val;
    if(val < cfg.min) val = cfg.min;
    if(val > cfg.max) val = cfg.max;
    return val;
  }

  soil::shape shape;
  float seed;
  FastNoiseLite source;
  sampler_t cfg;
};

};  // end of namespace soil

// Configuration Loading

#ifdef SOILLIB_IO_YAML

template<>
struct soil::io::yaml::cast<soil::noise::sampler_t> {
  static soil::noise::sampler_t As(soil::io::yaml& node){
    soil::noise::sampler_t sampler;
    sampler.frequency = node["frequency"].As<float>();
    sampler.octaves =  node["octaves"].As<int>();
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