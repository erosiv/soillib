#ifndef SOILLIB_LAYER_COMPUTED_NOISE
#define SOILLIB_LAYER_COMPUTED_NOISE

#include <soillib/core/types.hpp>
#include <soillib/core/index.hpp>
#include <soillib/core/buffer.hpp>

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

  noise(const soil::index index, const float seed):
  index{index},seed{seed}{
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

    return soil::indexselect(index.type(), [self=this]<typename T>() -> soil::buffer {

      if constexpr(std::same_as<typename T::vec_t, soil::ivec2>){

        auto index = self->index.as<T>();
        auto out = buffer_t<float>{index.elem()};

        for(const auto& pos: index.iter()){
          out[index.flatten(pos)] = self->operator()(pos);
        }

        return std::move(soil::buffer(std::move(out)));

      } else {
        throw std::invalid_argument("can't extract a full noise buffer from a non-2D index");
      }

    });

  }

private:

  inline float noise_impl(const soil::vec2 pos) {

    return soil::indexselect(index.type(), [self=this, pos]<typename T>() -> float {

      const T index = self->index.as<T>();
      const auto ext = soil::vec2{512, 512};//index.ext();
      const auto& cfg = self->cfg;

      float val = self->source.GetNoise(pos[0]/(float)ext[0], pos[1]/(float)ext[1], self->seed);
      
      // Clamp Value
      val = cfg.bias + cfg.scale * val;
      if(val < cfg.min) val = cfg.min;
      if(val > cfg.max) val = cfg.max;
      return val;

    });

  }

  soil::index index;
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