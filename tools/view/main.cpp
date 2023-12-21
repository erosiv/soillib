#include <TinyEngine/TinyEngine>
#include <TinyEngine/color>
#include <TinyEngine/camera>
#include <TinyEngine/image>

#include <soillib/util/pool.hpp>
#include <soillib/util/index.hpp>
#include <soillib/map/basic.hpp>
#include <soillib/io/tiff.hpp>
#include <soillib/io/png.hpp>

#include <soillib/model/surface.hpp>

#include "model.hpp"

struct cell {
  float height;
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

int main( int argc, char* args[] ) {

  if(argc < 2){
    std::cout<<"please specify input directory for dataset"<<std::endl;
    return 0;
  }
  std::string path = args[1];

  // Load Image Data

  soil::io::tiff height((path + "/height.tiff").c_str());
  //soil::io::tiff discharge((path + "/discharge.tiff").c_str());
  //soil::io::png normal((path + "/normal.png").c_str());
  //soil::io::png albedo((path + "/albedo.png").c_str());

  // Create Map

  const glm::ivec2 dim = glm::ivec2(height.width, height.height);
  world_t world(dim);

  // Fill Map

  for(auto [cell, pos]: world.map){
    cell.height = height[pos];
  }

  for(auto [cell, pos]: world.map){
    cell.normal = soil::surface::normal(world, pos);
  }

	Tiny::view.vsync = false;
	Tiny::window("soillib dataset viewer", 1200, 1000);			//Open Window

	cam::near = -1000.0f;
	cam::far = 1000.0f;
	cam::rot = 0.0f;
	cam::roty = 45.0f;
  cam::turnrate = 1.0f;
  cam::zoomrate *= 0.2f;
	cam::init(7, cam::ORTHO);
	cam::update();

	Tiny::event.handler = cam::handler;								//Event Handler
	Tiny::view.interface = [&](){ /* ... */ };				//No Interface

	Buffer positions, indices;												//Define Buffers
	construct(world.map, positions, indices);						    //Fill Buffers

	Model mesh({"in_Position"});					//Create Model with 2 Properties
	mesh.bind<glm::vec3>("in_Position", &positions);	//Bind Buffer to Property
	mesh.index(&indices);
  mesh.model = glm::scale(glm::mat4(1.0f), glm::vec3(0.1));
	mesh.model = glm::translate(mesh.model, glm::vec3(-world.map.dimension.x/2, -15.0, -world.map.dimension.y/2));
	Shader defaultShader({"shader/default.vs", "shader/default.fs"}, {"in_Position"});

  // Textures

 // Texture dischargeMap(image::make([&](const glm::ivec2 p){
 //   return glm::vec4(discharge[p]);
 // }, world.map.dimension));

  Texture normalMap(world.map.dimension.x, world.map.dimension.y, {GL_RGBA32F, GL_RGBA, GL_FLOAT});
  
  auto data = new glm::vec4[world.map.dimension.x*world.map.dimension.y];
  for(size_t x = 0; x < world.map.dimension.x; x++)
  for(size_t y = 0; y < world.map.dimension.y; y++){
    data[y*world.map.dimension.x + x] = glm::vec4(world.map.get(glm::ivec2(x, y))->normal, 0.0f);
  }

  glBindTexture(GL_TEXTURE_2D, normalMap.texture);

  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, world.map.dimension.x, world.map.dimension.y, 0, GL_RGBA, GL_FLOAT, data);

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glBindTexture(GL_TEXTURE_2D, 0);
  delete[] data;

  //Texture albedoMap(image::make([&](const glm::ivec2 p){
  //  return glm::vec4(albedo[p]);
  //}, world.map.dimension));

	Tiny::view.pipeline = [&](){											//Setup Drawing Pipeline

		Tiny::view.target(color::white);								//Target Screen

		defaultShader.use();														//Prepare Shader
		defaultShader.uniform("model", mesh.model);			//Set Model Matrix
    defaultShader.uniform("vp", cam::vp);						//View Projection Matrix
  //  defaultShader.texture("dischargeMap", dischargeMap);						//View Projection Matrix
    defaultShader.texture("normalMap", normalMap);            //View Projection Matrix
  //  defaultShader.texture("albedoMap", albedoMap);            //View Projection Matrix
    defaultShader.uniform("dimension", glm::vec2(world.map.dimension));						//View Projection Matrix
		mesh.render(GL_TRIANGLES);													//Render Model with Lines

	};

	Tiny::loop([&](){ //Autorotate Camera
	//	cam::pan(0.1f);
	});

	Tiny::quit();

	return 0;
}
