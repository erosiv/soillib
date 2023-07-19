int SEED = 10;
double scale = 30.0;

template<typename T>
void construct(T& map, Buffer& positions, Buffer& normals, Buffer& indices){
  //Fill the Buffers!

  std::vector<int> indbuf;
  std::vector<float> posbuf, norbuf;

  std::function<void(std::vector<float>&, glm::vec3)> add = [](std::vector<float>& v, glm::vec3 p){
    v.push_back(p.x);
    v.push_back(p.y);
    v.push_back(p.z);
  };

  for(auto [cell, pos]: map ){
    
    if(map.oob(pos + glm::ivec2(1)))
      continue;
    
    //Add to Position Vector
    glm::vec3 a = glm::vec3(pos.x, cell.height, pos.y);
    glm::vec3 b = glm::vec3(pos.x+1, map.get(pos + glm::ivec2(1, 0))->height, pos.y);
    glm::vec3 c = glm::vec3(pos.x, map.get(pos + glm::ivec2(0, 1))->height, pos.y+1);
    glm::vec3 d = glm::vec3(pos.x+1, map.get(pos + glm::ivec2(1, 1))->height, pos.y+1);

    //UPPER TRIANGLE

    //Add Indices
    indbuf.push_back(posbuf.size()/3+0);
    indbuf.push_back(posbuf.size()/3+2);
    indbuf.push_back(posbuf.size()/3+1);

    add(posbuf, a);
    add(posbuf, b);
    add(posbuf, d);

    glm::vec3 n1 = glm::cross(a-b, d-b);
    n1 = 0.5f*n1 + 0.5f;

    for(int i = 0; i < 3; i++)
      add(norbuf, n1);

      indbuf.push_back(posbuf.size()/3+0);
      indbuf.push_back(posbuf.size()/3+2);
      indbuf.push_back(posbuf.size()/3+1);

    add(posbuf, a);
    add(posbuf, d);
    add(posbuf, c);

      glm::vec3 n2 = glm::cross(a-d, c-d);
      n2 = 0.5f*n2 + 0.5f;

      for(int i = 0; i < 3; i++)
        add(norbuf, n2);

  }

  indices.fill(indbuf);
  positions.fill(posbuf);
  normals.fill(norbuf);

}
