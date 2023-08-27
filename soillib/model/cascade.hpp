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
  { t.add(glm::ivec2(), float(), M()) } -> std::same_as<float>;
};

template<cascade_matrix M, cascade_t<M> T>
void cascade(T& map, const glm::ivec2 ipos, const bool recascade = false){

  // Out-Of-Bounds Checking

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
  };

  static Point sn[8];
  int num = 0;

  for(auto& nn: n){

    glm::ivec2 npos = ipos + nn;

    if(map.oob(npos))
      continue;

    sn[num++] = { npos, map.height(npos), map.matrix(npos), length(glm::vec2(nn)) };

  }

  // Compute the Average Height (i.e. trivial non-spurious stable solution)


  float h_ave = map.height(ipos);
  for (int i = 0; i < num; ++i)
    h_ave += sn[i].h;
  h_ave /= (num+1.0f);

  for (int i = 0; i < num; ++i) {

    const auto& npos = sn[i].pos;

    // Let's just compute the height difference:

    {

     float diff = map.subheight(ipos) - map.subheight(npos);
      if(diff == 0)   //No Height Difference
        continue;
 
      const glm::ivec2& tpos = (diff > 0)?ipos:npos;
      const glm::ivec2& bpos = (diff > 0)?npos:ipos;
      const M& tmatrix = map.submatrix(tpos);
      const M& bmatrix = map.submatrix(bpos);

      float excess = 0.0f;
      excess = abs(diff) - sn[i].d*tmatrix.maxdiff();
      if(excess <= 0)  //No Excess
        continue;

      if(map.matrix(tpos).is_water && map.matrix(bpos).is_water){

        excess = abs(diff) - sn[i].d*tmatrix.maxdiff()*0.1;
        if(excess <= 0)  //No Excess
          continue;

      }

      float transfer = tmatrix.settling() * excess / 2.0f;
      transfer *= (1.0f - map.resistance(tpos));
      //transfer = map.maxremove(tpos, transfer);

      map.add(tpos,-transfer, tmatrix);
      map.add(bpos, transfer, tmatrix);

    }

  }

  if(!map.matrix(ipos).is_water)
    return;

  // if(map.matrix(ipos) - map.subheight(npos))

  for (int i = 0; i < num; ++i) {

    const auto& npos = sn[i].pos;

    {

      const float diff = map.height(ipos) - map.height(npos);

      const glm::ivec2& tpos = (diff > 0)?ipos:npos;
      const glm::ivec2& bpos = (diff > 0)?npos:ipos;
      const M& tmatrix = map.matrix(tpos);
      const M& bmatrix = map.matrix(bpos);

      float excess = 0.0f;
      excess = abs(diff) - sn[i].d*tmatrix.maxdiff();
      if(excess <= 0)  //No Excess
        continue;

      float transfer = tmatrix.settling() * excess / 2.0f;
      transfer *= (1.0f - map.resistance(tpos));
      transfer = map.maxremove(tpos, transfer);

      map.add(tpos,-transfer, tmatrix);
      map.add(bpos, transfer, tmatrix);

    }

  }

}

};  //  end of namespace phys
};  //  end of namespace soil

#endif
