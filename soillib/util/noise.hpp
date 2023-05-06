#ifndef SOILLIB_UTIL_NOISE
#define SOILLIB_UTIL_NOISE

namespace soil {

}; // end of namespace

#endif



    // Fill the Node Array

/*
  //  std::cout<<"Generating New World"<<std::endl;
  //  std::cout<<"Seed: "<<SEED<<std::endl;

  //  std::cout<<"... generating height ..."<<std::endl;

    static FastNoiseLite noise; //Noise System
    noise.SetNoiseType(FastNoiseLite::NoiseType_OpenSimplex2);
    noise.SetFractalType(FastNoiseLite::FractalType_FBm);


    for(auto& node: nodes){

      for(auto [cell, pos]: node.s){
        cell.height = 0.0f;
        cell.massflow = 0.0f;
      }


      // Add Gaussian

      for(auto [cell, pos]: node.s){
        vec2 p = vec2(node.pos+lodsize*pos)/vec2(tileres);
        vec2 c = vec2(node.pos+tileres/ivec2(4, 2))/vec2(tileres);
        float d = length(p-c);
        cell.height = exp(-d*d*tilesize*0.2);
      }

*/
/*

      // Add Layers of Noise

      float frequency = 1.0f;
      float scale = 0.6f;

      for(size_t o = 0; o < 8; o++){

        noise.SetFrequency(frequency);

        for(auto [cell, pos]: node.s){

          vec2 p = vec2(node.pos+lodsize*pos)/vec2(tileres);
          cell.height += scale*noise.GetNoise(p.x, p.y, (float)(SEED%10000));

        }

        frequency *= 2;
        scale *= 0.6;

      }

*/
/*

    }

    float min = 0.0f;
    float max = 0.0f;

    for(auto& node: nodes)
    for(auto [cell, pos]: node.s){
      min = (min < cell.height)?min:cell.height;
      max = (max > cell.height)?max:cell.height;
    }

    for(auto& node: nodes)
    for(auto [cell, pos]: node.s){
      cell.height = 0.5*((cell.height - min)/(max - min));
    }

*/
