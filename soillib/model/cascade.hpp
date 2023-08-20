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

    const M matrix = map.matrix(ipos);
    const auto& npos = sn[i].pos;
    sn[i].matrix = map.matrix(npos);







    if(matrix.is_water && sn[i].matrix.is_water){


      {

        //Full Height-Different Between Positions!
        float diff = map.height(ipos) - sn[i].h;
        if(diff == 0)   //No Height Difference
          continue;

        const M& tmatrix = (diff >= 0)?matrix:sn[i].matrix;
        const glm::ivec2& tpos = (diff >= 0)?ipos:npos;
        const glm::ivec2& bpos = (diff >= 0)?npos:ipos;

        //The Amount of Excess Difference!
        float excess = 0.0f;
        excess = abs(diff) - sn[i].d*tmatrix.maxdiff();
        if(excess <= 0)  //No Excess
          continue;

        //Actual Amount Transferred
        float transfer = tmatrix.settling() * excess / 2.0f;
        transfer = map.maxremove(tpos, transfer);

        //Cap by Maximum Transferrable Amount
        transfer = -map.add(tpos,-transfer, tmatrix);
        transfer = map.add(bpos, transfer, tmatrix);

      }

      auto& npos = sn[i].pos;

      //Full Height-Different Between Positions!
      float diff = map.subheight(ipos) - map.subheight(npos);
      if(diff == 0)   //No Height Difference
        continue;

      M tmatrix = matrix;
      tmatrix.is_water = false;

      const glm::ivec2& tpos = (diff >= 0)?ipos:npos;
      const glm::ivec2& bpos = (diff >= 0)?npos:ipos;

      //The Amount of Excess Difference!
      float excess = 0.0f;
      excess = abs(diff) - sn[i].d*0.5;
      if(excess <= 0)  //No Excess
        continue;

      //Actual Amount Transferred
      float transfer = tmatrix.settling() * excess / 2.0f;
     // transfer = map.maxremove(tpos, transfer);

      //Cap by Maximum Transferrable Amount
      map.map.top(tpos)->height -= transfer/map.config.scale;
      map.map.top(tpos)->below->height -= transfer/map.config.scale;

      map.map.top(bpos)->height += transfer/map.config.scale;
      map.map.top(bpos)->below->height += transfer/map.config.scale;

    //  transfer = -map.add(tpos,-transfer, tmatrix);
    //  transfer = map.add(bpos, transfer, tmatrix);

      if(recascade)
        cascade<M>(map, npos, false);

      continue;

    }


    // Same Matrix: Simple Cascade on Height

    if(sn[i].matrix.is_water == matrix.is_water) {

      //Full Height-Different Between Positions!
      float diff = map.height(ipos) - sn[i].h;
      if(diff == 0)   //No Height Difference
        continue;

      const M& tmatrix = (diff >= 0)?matrix:sn[i].matrix;
      const glm::ivec2& tpos = (diff >= 0)?ipos:npos;
      const glm::ivec2& bpos = (diff >= 0)?npos:ipos;

      //The Amount of Excess Difference!
      float excess = 0.0f;
      excess = abs(diff) - sn[i].d*tmatrix.maxdiff();
      if(excess <= 0)  //No Excess
        continue;

      //Actual Amount Transferred
      float transfer = tmatrix.settling() * excess / 2.0f;
      transfer = map.maxremove(tpos, transfer);

      //Cap by Maximum Transferrable Amount
      transfer = -map.add(tpos,-transfer, tmatrix);
      transfer = map.add(bpos, transfer, tmatrix);

      if(recascade)
        cascade<M>(map, npos, false);

      continue;

    }




    if(!matrix.is_water && sn[i].matrix.is_water){

      auto& npos = sn[i].pos;

      //Full Height-Different Between Positions!
      float diff;

      if(map.height(ipos) >= map.height(npos)){
        diff = map.height(ipos) - map.subheight(npos);
      }

      else {
        diff = map.height(ipos) - map.height(npos);
      }

      if(diff == 0)   //No Height Difference
        continue;

      const M& tmatrix = (diff >= 0)?matrix:sn[i].matrix;
      const glm::ivec2& tpos = (diff >= 0)?ipos:npos;
      const glm::ivec2& bpos = (diff >= 0)?npos:ipos;

      //The Amount of Excess Difference!
      float excess = 0.0f;
      excess = abs(diff) - sn[i].d*tmatrix.maxdiff();
      if(excess <= 0)  //No Excess
        continue;

      //Actual Amount Transferred
      float transfer = tmatrix.settling() * excess / 2.0f;
      transfer = map.maxremove(tpos, transfer);

      //Cap by Maximum Transferrable Amount
      transfer = -map.add(tpos,-transfer, tmatrix);
      transfer = map.add(bpos, transfer, tmatrix);

      if(recascade)
        cascade<M>(map, npos, false);

      continue;      

    }

    if(matrix.is_water && !sn[i].matrix.is_water){

      auto& npos = sn[i].pos;

      //Full Height-Different Between Positions!
      float diff;

      if(map.height(ipos) > map.height(npos)){
        diff = map.height(ipos) - map.height(npos);
      }

      else {
        diff = map.subheight(ipos) - map.height(npos);
      }

      if(diff == 0)   //No Height Difference
        continue;

      const M& tmatrix = (diff >= 0)?matrix:sn[i].matrix;
      const glm::ivec2& tpos = (diff >= 0)?ipos:npos;
      const glm::ivec2& bpos = (diff >= 0)?npos:ipos;

      //The Amount of Excess Difference!
      float excess = 0.0f;
      excess = abs(diff) - sn[i].d*tmatrix.maxdiff();
      if(excess <= 0)  //No Excess
        continue;

      //Actual Amount Transferred
      float transfer = tmatrix.settling() * excess / 2.0f;
      transfer = map.maxremove(tpos, transfer);

      //Cap by Maximum Transferrable Amount
      transfer = -map.add(tpos,-transfer, tmatrix);
      transfer = map.add(bpos, transfer, tmatrix);

      if(recascade)
        cascade<M>(map, npos, false);

      continue;      

    }

  }

}

};  //  end of namespace phys
};  //  end of namespace soil

#endif
