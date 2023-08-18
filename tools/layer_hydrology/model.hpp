template<typename T>
void construct(T& world, Buffer& positions, Buffer& indices){
  //Fill the Buffers!

  std::vector<int> indbuf;
  std::vector<float> posbuf;

  std::function<void(std::vector<float>&, glm::vec3)> add = [](std::vector<float>& v, glm::vec3 p){
    v.push_back(p.x);
    v.push_back(p.y);
    v.push_back(p.z);
  };

  for(auto [cell, pos]: world.map ){
    
    if(world.map.oob(pos + glm::ivec2(1)))
      continue;
    
    glm::vec3 a, b, c, d;

    //Add to Position Vector

    if(world.matrix(pos + glm::ivec2(0, 0)).is_water)
      a = glm::vec3(pos.x, world.config.scale*world.map.top(pos + glm::ivec2(0, 0))->below->height, pos.y);
    else
      a = glm::vec3(pos.x, world.height(pos + glm::ivec2(0, 0)), pos.y);

    if(world.matrix(pos + glm::ivec2(1, 0)).is_water)
      b = glm::vec3(pos.x+1, world.config.scale*world.map.top(pos + glm::ivec2(1, 0))->below->height, pos.y);
    else
      b = glm::vec3(pos.x+1, world.height(pos + glm::ivec2(1, 0)), pos.y);

    if(world.matrix(pos + glm::ivec2(0, 1)).is_water)
      c = glm::vec3(pos.x, world.config.scale*world.map.top(pos + glm::ivec2(0, 1))->below->height, pos.y+1);
    else
      c = glm::vec3(pos.x, world.height(pos + glm::ivec2(0, 1)), pos.y+1);

    if(world.matrix(pos + glm::ivec2(1, 1)).is_water)
      d = glm::vec3(pos.x+1, world.config.scale*world.map.top(pos + glm::ivec2(1, 1))->below->height, pos.y+1);
    else
      d = glm::vec3(pos.x+1, world.height(pos + glm::ivec2(1, 1)), pos.y+1);

    //UPPER TRIANGLE

    //Add Indices
    indbuf.push_back(posbuf.size()/3+0);
    indbuf.push_back(posbuf.size()/3+2);
    indbuf.push_back(posbuf.size()/3+1);

    add(posbuf, a);
    add(posbuf, b);
    add(posbuf, d);
    
    indbuf.push_back(posbuf.size()/3+0);
    indbuf.push_back(posbuf.size()/3+2);
    indbuf.push_back(posbuf.size()/3+1);

    add(posbuf, a);
    add(posbuf, d);
    add(posbuf, c);

  }

  
  for(auto [cell, pos]: world.map ){
    
    if(world.map.oob(pos + glm::ivec2(1)))
      continue;
    
    //Add to Position Vector
    glm::vec3 a = glm::vec3(pos.x, world.height(pos + glm::ivec2(0, 0)), pos.y);
    glm::vec3 b = glm::vec3(pos.x+1, world.height(pos + glm::ivec2(1, 0)), pos.y);
    glm::vec3 c = glm::vec3(pos.x, world.height(pos + glm::ivec2(0, 1)), pos.y+1);
    glm::vec3 d = glm::vec3(pos.x+1, world.height(pos + glm::ivec2(1, 1)), pos.y+1);

    //UPPER TRIANGLE

    //Add Indices
    indbuf.push_back(posbuf.size()/3+0);
    indbuf.push_back(posbuf.size()/3+2);
    indbuf.push_back(posbuf.size()/3+1);

    add(posbuf, a);
    add(posbuf, b);
    add(posbuf, d);
    
    indbuf.push_back(posbuf.size()/3+0);
    indbuf.push_back(posbuf.size()/3+2);
    indbuf.push_back(posbuf.size()/3+1);

    add(posbuf, a);
    add(posbuf, d);
    add(posbuf, c);

  }
  

  indices.fill(indbuf);
  positions.fill(posbuf);

}
