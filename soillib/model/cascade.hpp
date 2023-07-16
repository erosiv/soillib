#ifndef SOILLIB_PHYSICS_CASCADE
#define SOILLIB_PHYSICS_CASCADE

#include <soillib/soillib.hpp>
#include <algorithm>

namespace soil {
namespace phys {

// Cascadable Map Constraints

template<typename T>
concept cascade_t = requires(T t){
  // Measurement Methods
  { t.height(glm::ivec2()) } -> std::same_as<float>;
  { t.oob(glm::ivec2()) } -> std::same_as<bool>;
  // Add Method
  { t.add(glm::ivec2(), float()) } -> std::same_as<void>;
};

// Cascading Parameters

struct cascade_c {
  static float maxdiff;
  static float settling;
};

float cascade_c::maxdiff = 0.01f;
float cascade_c::settling = 1.0f;

template<cascade_t T>
void cascade(T& map, const glm::ivec2 ipos){

  // Get Non-Out-of-Bounds Neighbors

  static const glm::ivec2 n[] = {
    glm::ivec2(-1, -1),
    glm::ivec2(-1,  0),
    glm::ivec2(-1,  1),
    glm::ivec2( 0, -1),
    glm::ivec2( 0,  1),
    glm::ivec2( 1, -1),
    glm::ivec2( 1,  0),
    glm::ivec2( 1,  1)
  };

  struct Point {
    glm::ivec2 pos;
    float h;
    float d;
  };

  static Point sn[8];
  int num = 0;

  for(auto& nn: n){

    glm::ivec2 npos = ipos + nn;

    if(map.oob(npos))
      continue;

    sn[num++] = { npos, map.height(npos), length(glm::vec2(nn)) };

  }

  // Compute the Average Height (i.e. trivial non-spurious stable solution)

  float h_ave = 0.0f;
  for (int i = 0; i < num; ++i)
    h_ave += sn[i].h;
  h_ave /= (float)num;

  for (int i = 0; i < num; ++i) {

    auto& npos = sn[i].pos;

    //Full Height-Different Between Positions!
    float diff = h_ave - sn[i].h;
    if(diff == 0)   //No Height Difference
      continue;

    //The Amount of Excess Difference!
    float excess = 0.0f;
    excess = abs(diff) - sn[i].d*cascade_c::maxdiff;
    if(excess <= 0)  //No Excess
      continue;

    //Actual Amount Transferred
    float transfer = cascade_c::settling * excess / 2.0f;

    //Cap by Maximum Transferrable Amount
    if(diff > 0) {
      map.add(ipos,-transfer);
      map.add(npos, transfer);
    }
    else {
      map.add(ipos, transfer);
      map.add(npos,-transfer);
    }

  }

}

};  //  end of namespace phys
};  //  end of namespace soil

#endif
