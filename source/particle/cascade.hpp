#ifndef SOILLIB_PARTICLE_CASCADE
#define SOILLIB_PARTICLE_CASCADE

#include <soillib/soillib.hpp>
#include <soillib/util/array.hpp>
#include <soillib/layer/constant.hpp>

#include <soillib/matrix/matrix.hpp>

#include <algorithm>

//! Note: This will be refactored as a type of 
//! Rock or Sediment Particle. That way the
//! cascading / avalanching motion can be made
//! more dynamic / precise and potentially also
//! tracked. For now, the implementation will
//! remain grid-based but in the future it will
//! be full interactive particle with thermal
//! erosion based spawn rates.

namespace soil {

struct cascade_model_t {

  using matrix_t = soil::matrix::singular;

  soil::shape shape;
  soil::array& height;
  soil::constant maxdiff;
  soil::constant settling;

  void add(const size_t index, const float value, const matrix_t matrix){
    auto _height = std::get<soil::array_t<float>>(this->height._array);
    _height[index] += value;// / 80.0f;
  }

};

void cascade(soil::cascade_model_t& model, const glm::ivec2 ipos){

  using model_t = soil::cascade_model_t;
  using matrix_t = soil::matrix::singular;

  if(model.shape.oob(ipos))
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
    matrix_t matrix;
    float d;
  } static sn[8];

  int num = 0;

  for(auto& nn: n){

    glm::ivec2 npos = ipos + nn;

    if(model.shape.oob(npos))
      continue;

    const size_t index = model.shape.flat(npos);
    const float height = std::get<float>(model.height[index]);
    sn[num++] = { npos, height, matrix_t{}, length(glm::vec2(nn)) };

  }

  // Local Matrix, Target Height

  const matrix_t matrix{};// = map.matrix(ipos);

  const size_t index = model.shape.flat(ipos);
  float h_ave = std::get<float>(model.height[index]);
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
    const matrix_t& tmatrix = (diff > 0)?matrix:sn[i].matrix;

    const size_t tindex = model.shape.flat(tpos);
    const size_t bindex = model.shape.flat(bpos);

    const float maxdiff = model.maxdiff.template operator()<float>(tindex);
    const float settling = model.settling.template operator()<float>(tindex);

    //The Amount of Excess Difference!
    float excess = 0.0f;
    excess = abs(diff) - sn[i].d*maxdiff;
    if(excess <= 0)  //No Excess
      continue;

    //Actual Amount Transferred
    float transfer = settling * excess / 2.0f;
    model.add(tindex, -transfer, tmatrix);
    model.add(bindex,  transfer, tmatrix);

  }

}

};  //  end of namespace soil

#endif
