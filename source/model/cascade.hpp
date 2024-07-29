#ifndef SOILLIB_PHYSICS_CASCADE
#define SOILLIB_PHYSICS_CASCADE

#include <soillib/soillib.hpp>
#include <algorithm>

namespace soil {
namespace phys {

// Cascadable Matrix Constraints

template<typename M>
concept cascade_matrix = requires(M m){
  { m.maxdiff() } -> std::same_as<float>;
  { m.settling() } -> std::same_as<float>;
};

// Cascadable Map Constraints

template<typename T, typename M>
concept cascade_t = requires(T t){
  { t.height(glm::ivec2()) } -> std::same_as<float>;
  { t.oob(glm::ivec2()) } -> std::same_as<bool>;
  { t.matrix(glm::ivec2()) } -> std::same_as<M>;
  { t.add(glm::ivec2(), float(), M()) } -> std::same_as<void>;
};

template<cascade_matrix M, cascade_t<M> T>
void cascade(T& map, const glm::ivec2 ipos){

  if(map.oob(ipos))
    return;

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
    M matrix;
    float d;
  } static sn[8];

  int num = 0;

  for(auto& nn: n){

    glm::ivec2 npos = ipos + nn;

    if(map.oob(npos))
      continue;

    sn[num++] = { npos, map.height(npos), map.matrix(npos), length(glm::vec2(nn)) };

  }

  // Local Matrix, Target Height

  const M matrix = map.matrix(ipos);

  float h_ave = map.height(ipos);
  for (int i = 0; i < num; ++i)
    h_ave += sn[i].h;
  h_ave /= (float)(num+1);

  for (int i = 0; i < num; ++i) {

    //Full Height-Different Between Positions!
    float diff = h_ave - sn[i].h;
    if(diff == 0)   //No Height Difference
      continue;

    const glm::ivec2& tpos = (diff > 0)?ipos:sn[i].pos;
    const glm::ivec2& bpos = (diff > 0)?sn[i].pos:ipos;
    const M& tmatrix = (diff > 0)?matrix:sn[i].matrix;

    //The Amount of Excess Difference!
    float excess = 0.0f;
    excess = abs(diff) - sn[i].d*tmatrix.maxdiff();
    if(excess <= 0)  //No Excess
      continue;

    //Actual Amount Transferred
    float transfer = tmatrix.settling() * excess / 2.0f;
    map.add(tpos,-transfer, tmatrix);
    map.add(bpos, transfer, tmatrix);

  }

}

};  //  end of namespace phys
};  //  end of namespace soil

#endif
