#ifndef SOILLIB_PHYSICS_CASCADE
#define SOILLIB_PHYSICS_CASCADE

namespace soil {
namespace phys {

struct cascade_c {
  static float maxdiff;
  static float settling;
};

float cascade_c::maxdiff = 0.01f;
float cascade_c::settling = 0.8f;

template<typename T>
void cascade(T& map, const ivec2 ipos){

  // Get Non-Out-of-Bounds Neighbors

  static const ivec2 n[] = {
    ivec2(-1, -1),
    ivec2(-1,  0),
    ivec2(-1,  1),
    ivec2( 0, -1),
    ivec2( 0,  1),
    ivec2( 1, -1),
    ivec2( 1,  0),
    ivec2( 1,  1)
  };

  struct Point {
    ivec2 pos;
    float h;
    float d;
  };

  static Point sn[8];
  int num = 0;

  for(auto& nn: n){

    ivec2 npos = ipos + nn;

    if(map.oob(npos))
      continue;

    sn[num++] = { npos, map.get(npos)->height, length(vec2(nn)) };

  }

  //Iterate over all sorted Neighbors

  sort(std::begin(sn), std::begin(sn) + num, [&](const Point& a, const Point& b){
    return a.h < b.h;
  });

  for (int i = 0; i < num; ++i) {

    auto& npos = sn[i].pos;

    //Full Height-Different Between Positions!
    float diff = map.get(ipos)->height - sn[i].h;
    if(diff == 0)   //No Height Difference
      continue;

      //The Amount of Excess Difference!
    float excess = 0.0f;
    if(sn[i].h > 0.1){
      excess = abs(diff) - sn[i].d*cascade_c::maxdiff;
    } else {
      excess = abs(diff);
    }

    if(excess <= 0)  //No Excess
      continue;

    //Actual Amount Transferred
    float transfer = cascade_c::settling * excess / 2.0f;

    //Cap by Maximum Transferrable Amount
    if(diff > 0) {
      map.get(ipos)->height -= transfer;
      map.get(npos)->height += transfer;
    }
    else {
      map.get(ipos)->height += transfer;
      map.get(npos)->height -= transfer;
    }

  }

}

};  //  end of namespace phys
};  //  end of namespace soil

#endif
