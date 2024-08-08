#ifndef SOILLIB_LAYER_NORMAL
#define SOILLIB_LAYER_NORMAL

#include <soillib/soillib.hpp>
#include <soillib/util/array.hpp>
#include <soillib/layer/layer.hpp>

// 

/*
Normal Node: Takes a floating point node
and returns a vec3 normal vector node.

Should be indexable but also fully cachable.
Basically this will make this almost differentiable...
*/

/*
In theory, I could have a generic node that just takes a lambda right?
Let's not do that for now though. In theory this could be done later
for proper interactive nodes w. std::function?
*/

namespace soil {

namespace {

glm::vec2 gradient_detailed(const soil::array& array, glm::ivec2 p){

  // Generate the Finite Difference Samples

  struct Point {
    glm::ivec2 pos;
    bool oob = true;
    float height;
  } px[5], py[5];

  px[0].pos = p + glm::ivec2(-2, 0);
  px[1].pos = p + glm::ivec2(-1, 0);
  px[2].pos = p + glm::ivec2( 0, 0);
  px[3].pos = p + glm::ivec2( 1, 0);
  px[4].pos = p + glm::ivec2( 2, 0);

  py[0].pos = p + glm::ivec2( 0,-2);
  py[1].pos = p + glm::ivec2( 0,-1);
  py[2].pos = p + glm::ivec2( 0, 0);
  py[3].pos = p + glm::ivec2( 0, 1);
  py[4].pos = p + glm::ivec2( 0, 2);

  // if(map.shape())

  auto _array = std::get<soil::array_t<float>>(array);
  auto _shape = std::get<soil::shape_t<2>>(_array.shape());

  /*
  auto shape = [](const soil::array& array){
    return std::visit([](auto&& args){
      return args.shape();
    }, array);
  };
  */

  // std::cout<<p.x<<" "<<p.y<<std::endl;


  auto oob = [&](const glm::ivec2 pos) -> bool {
    return _shape.oob(pos);
  };

  auto sample = [&](const glm::ivec2 pos) -> float {
    const size_t index = _shape.flat({(size_t)pos.x, (size_t)pos.y});

    //std::cout<<pos.x<<" "<<pos.y<<" "<<index<<std::endl;

    return _array[index];
  };

  for(size_t i = 0; i < 5; i++){
    if(!oob(px[i].pos)){
      px[i].oob = false;
      px[i].height = sample(px[i].pos);

     // std::cout<<i<<" "<<px[i].height<<std::endl;

    }
  }
  
  for(size_t i = 0; i < 5; i++){
    if(!oob(py[i].pos)){
      py[i].oob = false;
      py[i].height = sample(py[i].pos);


     // std::cout<<i<<" "<<py[i].height<<std::endl;

    }
  }

  // Compute Gradient

  glm::vec2 g = glm::vec2(0, 0);

  // X-Element
  /*
  
  if(!px[0].oob && !px[4].oob)
    g.x = (1.0f*px[0].height - 8.0f*px[1].height + 8.0f*px[3].height - 1.0f*px[4].height)/12.0f;

  else if(!px[0].oob && !px[3].oob)
    g.x = (1.0f*px[0].height - 6.0f*px[1].height + 3.0f*px[2].height + 2.0f*px[3].height)/6.0f;

  else if(!px[0].oob && !px[2].oob)
    g.x = (1.0f*px[0].height - 4.0f*px[1].height + 3.0f*px[2].height)/2.0f;

  else if(!px[1].oob && !px[4].oob)
    g.x = (-2.0f*px[1].height - 3.0f*px[2].height + 6.0f*px[3].height - 1.0f*px[4].height)/6.0f;

  else if(!px[2].oob && !px[4].oob)
    g.x = (-3.0f*px[2].height + 4.0f*px[3].height - 1.0f*px[4].height)/2.0f;

  else */ if(!px[1].oob && !px[3].oob)
    g.x = (-1.0f*px[1].height + 1.0f*px[3].height)/2.0f;

  else if(!px[2].oob && !px[3].oob)
    g.x = (-1.0f*px[2].height + 1.0f*px[3].height)/1.0f;

  else  if(!px[1].oob && !px[2].oob)
    g.x = (-1.0f*px[1].height + 1.0f*px[2].height)/1.0f;

  // Y-Element

  /* if(!py[0].oob && !py[4].oob)
    g.y = (1.0f*py[0].height - 8.0f*py[1].height + 8.0f*py[3].height - 1.0f*py[4].height)/12.0f;

  else if(!py[0].oob && !py[3].oob)
    g.y = (1.0f*py[0].height - 6.0f*py[1].height + 3.0f*py[2].height + 2.0f*py[3].height)/6.0f;

  else if(!py[0].oob && !py[2].oob)
    g.y = (1.0f*py[0].height - 4.0f*py[1].height + 3.0f*py[2].height)/2.0f;

  else if(!py[1].oob && !py[4].oob)
    g.y = (-2.0f*py[1].height - 3.0f*py[2].height + 6.0f*py[3].height - 1.0f*py[4].height)/6.0f;

  else if(!py[2].oob && !py[4].oob)
    g.y = (-3.0f*py[2].height + 4.0f*py[3].height - 1.0f*py[4].height)/2.0f;

  else */ if(!py[1].oob && !py[3].oob)
    g.y = (-1.0f*py[1].height + 1.0f*py[3].height)/2.0f;

  else if(!py[2].oob && !py[3].oob)
    g.y = (-1.0f*py[2].height + 1.0f*py[3].height)/1.0f;

  else  if(!py[1].oob && !py[2].oob)
    g.y = (-1.0f*py[1].height + 1.0f*py[2].height)/1.0f;

  return g;

}

// Surface Normal from Surface Gradient

//template<surface_t T>
glm::vec3 __normal(const soil::array& array, glm::ivec2 p){

  const glm::vec2 g = gradient_detailed(array, p);
  glm::vec3 n = glm::vec3(-g.x, 1.0f, -g.y);

  if(length(n) > 0)
    n = normalize(n);
  return n;

}

}











struct normal: layer<array_t<float>, array_t<fvec3>> {

  using layer::layer;
  using layer::in_t;
  using layer::out_t;
  using layer::in;

  using layer_t = layer<array_t<float>, array_t<fvec3>>;

  /*

  // whoat goes on where?

  normal(){



  }

  // full operation is just a 
  // basically the input could also be another node though...
  // is this relevant? not sure right now. It could be e.g. warped.
  // should I cache the in-node upon construction?

  */

private:
  
  /*
  static out_t operator()(const in_t& in){
    
    soil::shape shape = in.shape();
    out_t out = out_t{shape};
    return std::move(out);

  }
  */

public:
  out_t operator()(){
  //  return normal::operator()(this->in);

    soil::shape shape = in.shape();
    out_t out = out_t{shape};

    /*
    auto _shape = std::get<soil::shape_t<2>>(shape);
    for(const auto& pos: _shape.iter()){
      const size_t index = _shape.flat(pos);
      const float value = (in[index] - 607.7095) / (992.8845 - 607.7095);
      out[index] = {value, value, value};
    }
    */

    auto _shape = std::get<soil::shape_t<2>>(shape);
    for(const auto& pos: _shape.iter()){

      const size_t index = _shape.flat(pos);
      glm::vec3 n = __normal(in, glm::ivec2(pos[0], pos[1]));
      n = glm::vec3(n.x, -n.z, n.y);
      n = 0.5f*n + 0.5f;
      out[index] = {n.x, n.y, n.z};
    }

    return std::move(out);
  }

};

} // end of namespace soil

#endif