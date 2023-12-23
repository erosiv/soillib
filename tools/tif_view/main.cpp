#include <TinyEngine/TinyEngine>
#include <TinyEngine/color>
#include <TinyEngine/camera>
#include <TinyEngine/image>

#include <soillib/util/pool.hpp>
#include <soillib/util/index.hpp>
#include <soillib/map/basic.hpp>
#include <soillib/io/geotiff.hpp>
#include <soillib/io/png.hpp>

#include <soillib/model/surface.hpp>

#include <limits>
#include <filesystem>

#include "vertexpool.cpp"
#include "model.hpp"

struct cell {
  float height = 0.0f;
  glm::vec3 normal;
};

struct world_t {

  const glm::ivec2 dim;
  soil::map::basic<cell> map;

  world_t(const glm::ivec2 dim)
    :dim(dim),map(dim){}

  const inline bool oob(glm::vec2 p){
    return map.oob(p);
  }

  const inline float height(glm::vec2 p){
    return map.get(p)->height;
  }

};

typedef float value_t;
typedef soil::io::geotiff<value_t> geovalue_t;

struct dem_t {
  std::string name;
  geovalue_t img;
};

int main( int argc, char* args[] ) {

  // Directory Path

  if(argc < 2){
    std::cout<<"please specify input directory for dataset"<<std::endl;
    return 0;
  }
  std::string path = args[1];

  // Load Image Data

  std::vector<dem_t> images;
  for(const auto& entry: std::filesystem::directory_iterator(path)){
    geovalue_t dem;
    dem.meta(entry.path().c_str());
    /*
    if(dem.coords[1].x <= 23749.8) continue;
    if(dem.coords[1].x >= 38750.2) continue;
    if(dem.coords[1].y <= 294000) continue;
    if(dem.coords[1].y >= 309001) continue;
    */
    images.push_back({
      entry.path(),
      dem
    });
  }

  std::cout<<"Images: "<<images.size()<<std::endl;

  // Compute Bounding Box

  glm::vec2 min = glm::vec2(std::numeric_limits<value_t>::max());
  glm::vec2 max = glm::vec2(std::numeric_limits<value_t>::min());

  for(auto& image: images){
    image.img.read(image.name.c_str());

    const glm::vec2 imdim = glm::vec2(image.img.width, image.img.height);

    const glm::vec2 immin = glm::vec2(image.img.coords[1]);
    const glm::vec2 immax = glm::vec2(image.img.coords[1]) + glm::vec2(image.img.scale)*imdim;

    if(min.x > immin.x) min.x = immin.x;
    if(min.y > immin.y) min.y = immin.y;
    if(max.x < immax.x) max.x = immax.x;
    if(max.y < immax.y) max.y = immax.y;
  }

  std::cout<<"Bounds: ";
  std::cout<<"("<<min.x<<" "<<min.y<<"), ";
  std::cout<<"("<<max.x<<" "<<max.y<<")"<<std::endl;

  // Create Map

  const int downscale = 20;
  glm::ivec2 mapdim = (max - min) / glm::vec2(0.5) / glm::vec2(downscale);

  world_t world(mapdim);
  for(auto [cell, pos]: world.map){
    cell.height = 0.0f;
  }

  std::cout<<"Map: ";
  std::cout<<"("<<mapdim.x<<" "<<mapdim.y<<")"<<std::endl;

  // Load Images, Fill w. Height

  for(auto& image: images){

    const int width = image.img.width/downscale;
    const int height = image.img.height/downscale;

    for(size_t x = 0; x < width; x++)
    for(size_t y = 0; y < height; y++){
      glm::vec2 ipos = downscale*glm::ivec2(x, y);
      float test = image.img[downscale*glm::ivec2(x, height - 1 - y)];

      glm::vec2 wmin = image.img.coords[1];
      glm::vec2 wpos = wmin + ipos * glm::vec2(image.img.scale);
      glm::vec2 mpos = (wpos - min) / glm::vec2(image.img.scale) / glm::vec2(downscale);
      mpos.y = mapdim.y - 1 - mpos.y;
      world.map.get(mpos)->height = test;
    }
  }

  float hmax = std::numeric_limits<value_t>::min();
  float hmin = std::numeric_limits<value_t>::max();

  for(auto [cell, pos]: world.map){

    //std::cout<<cell.height<<std::endl;

    //if(cell.height = -999)
    //  continue;

    if(cell.height < hmin) hmin = cell.height;
    if(cell.height > hmax) hmax = cell.height;
  }

  for(auto [cell, pos]: world.map){
    cell.height -= hmin;
  }

  // Fill Normal
  for(auto [cell, pos]: world.map){
    cell.normal = soil::surface::normal(world, pos);
  }

  for(auto [cell, pos]: world.map){
    cell.height /= downscale * 0.5;
  }

	Tiny::view.vsync = false;
	Tiny::window("soillib geotiff viewer", 1200, 1000);			//Open Window

	cam::near = -1200.0f;
	cam::far = 1200.0f;
	cam::rot = 0.0f;
	cam::roty = 90.0f;
  cam::turnrate = 1.0f;
  cam::moverate *= 3.0f;
  cam::zoomrate *= 0.5f;
	cam::init(0.75, cam::ORTHO);
	cam::update();

	Tiny::event.handler = cam::handler;								//Event Handler
	Tiny::view.interface = [&](){ };				//No Interface

  /*
	Buffer positions, indices;												//Define Buffers
	construct(world, positions, indices);						    //Fill Buffers
  */

  Vertexpool<Vertex> vertexpool(mapdim.x*mapdim.y*4, 1);
  auto section = vertexpool.section(mapdim.x*mapdim.y*4);
  for(size_t x = 0; x < mapdim.x-1; x++)
  for(size_t y = 0; y < mapdim.y-1; y++){

    glm::ivec2 p = glm::ivec2(x, y);
    size_t ind = p.x*mapdim.y + p.y;

    if(world.map.get(p)->height == 0)
      continue;

    if(world.map.get(p+glm::ivec2(0, 1))->height == 0)
      continue;

    if(world.map.get(p+glm::ivec2(1, 1))->height == 0)
      continue;

    if(world.map.get(p+glm::ivec2(1, 1))->height == 0)
      continue;

    vertexpool.fill(section, 4*ind+0,
      glm::vec3(p.x, world.map.get(p)->height, p.y),
      world.map.get(p)->normal
    );

    vertexpool.fill(section, 4*ind+1,
      glm::vec3(p.x, world.map.get(p+glm::ivec2(0, 1))->height, p.y+1),
      world.map.get(p)->normal
    );

    vertexpool.fill(section, 4*ind+2,
      glm::vec3(p.x+1, world.map.get(p+glm::ivec2(1, 0))->height, p.y),
      world.map.get(p)->normal
    );

    vertexpool.fill(section, 4*ind+3,
      glm::vec3(p.x+1, world.map.get(p+glm::ivec2(1, 1))->height, p.y+1),
      world.map.get(p)->normal
    );

  }


  vertexpool.update();

  /*
	Model mesh({"in_Position"});					//Create Model with 2 Properties
	mesh.bind<glm::vec3>("in_Position", &positions);	//Bind Buffer to Property
	mesh.index(&indices);
	mesh.model = glm::translate(glm::mat4(1.0f), glm::vec3(-world.map.dimension.x/2, -15.0, -world.map.dimension.y/2));
  */

  glm::mat4 model = glm::translate(glm::mat4(1.0f), glm::vec3(-world.map.dimension.x/2, -15.0, -world.map.dimension.y/2));



	Shader defaultShader({"shader/default.vs", "shader/default.fs"}, {"in_Position", "in_Normal"});

  // Textures

  Texture normalMap(world.map.dimension.x, world.map.dimension.y, {GL_RGBA32F, GL_RGBA, GL_FLOAT});
  
  auto data = new glm::vec4[world.map.dimension.x*world.map.dimension.y];
  for(size_t x = 0; x < world.map.dimension.x; x++)
  for(size_t y = 0; y < world.map.dimension.y; y++){
    data[y*world.map.dimension.x + x] = glm::vec4(world.map.get(glm::ivec2(x, y))->normal, 1.0f);
  }

  glBindTexture(GL_TEXTURE_2D, normalMap.texture);

  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, world.map.dimension.x, world.map.dimension.y, 0, GL_RGBA, GL_FLOAT, data);

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glBindTexture(GL_TEXTURE_2D, 0);
  delete[] data;

	Tiny::view.pipeline = [&](){											//Setup Drawing Pipeline

		Tiny::view.target(color::white);								//Target Screen

		defaultShader.use();														//Prepare Shader
		defaultShader.uniform("model", model);			//Set Model Matrix
    defaultShader.uniform("vp", cam::vp);						//View Projection Matrix
    defaultShader.texture("normalMap", normalMap);            //View Projection Matrix
    defaultShader.uniform("dimension", glm::vec2(world.map.dimension));						//View Projection Matrix
    vertexpool.render(GL_TRIANGLES);                          //Render Model with Lines
    //vertexpool.render(GL_LINES);                          //Render Model with Lines

	};

	Tiny::loop([&](){ //Autorotate Camera
	//	cam::pan(0.1f);
	});

	Tiny::quit();

	return 0;
}
