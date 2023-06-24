int SEED = 10;
double scale = 30.0;
double heightmap[GRIDSIZE][GRIDSIZE] = {0.0};
glm::vec2 dim = glm::vec2(GRIDSIZE);

void construct(Buffer& positions, Buffer& normals, Buffer& indices){
  //Fill the Buffers!

  std::vector<int> indbuf;
  std::vector<float> posbuf, norbuf;

  std::function<void(std::vector<float>&, glm::vec3)> add = [](std::vector<float>& v, glm::vec3 p){
    v.push_back(p.x);
    v.push_back(p.y);
    v.push_back(p.z);
  };

  //Loop over all positions and add the triangles!
  for(int i = 0; i < dim.x-1; i++){
    for(int j = 0; j < dim.y-1; j++){

      //Add to Position Vector
      glm::vec3 a = glm::vec3(i, heightmap[i][j], j);
      glm::vec3 b = glm::vec3(i+1, heightmap[i+1][j], j);
      glm::vec3 c = glm::vec3(i, heightmap[i][j+1], j+1);
      glm::vec3 d = glm::vec3(i+1, heightmap[i+1][j+1], j+1);

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
  }

  indices.fill(indbuf);
  positions.fill(posbuf);
  normals.fill(norbuf);

}
