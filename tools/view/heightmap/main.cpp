#include <TinyEngine/TinyEngine>
#include <TinyEngine/color>
#include <TinyEngine/camera>

#include <soillib/io/tiff.hpp>

#define GRIDSIZE 1024

#include "model.h"

int main( int argc, char* args[] ) {

  	soil::io::tiff image("height.tiff");

  	for(int i = 0; i < dim.x; i++)
	for(int j = 0; j < dim.y; j++){
      	heightmap[i][j] = 80.0f*image[glm::ivec2(i, j)];
  	}

	Tiny::view.vsync = false;
	Tiny::window("Heightmap Render", 1200, 800);			//Open Window

	cam::near = -500.0f;
	cam::far = 500.0f;
	cam::rot = 45.0f;
	cam::roty = 45.0f;
	cam::init(10, cam::ORTHO);
	cam::update();

	Tiny::event.handler = cam::handler;								//Event Handler
	Tiny::view.interface = [&](){ /* ... */ };				//No Interface

	Buffer positions, normals;												//Define Buffers
	Buffer indices;
	construct(positions, normals, indices);						//Call algorithm to fill buffers

	Model mesh({"in_Position", "in_Normal"});					//Create Model with 2 Properties
	mesh.bind<glm::vec3>("in_Position", &positions);	//Bind Buffer to Property
	mesh.bind<glm::vec3>("in_Normal", &normals);
	mesh.index(&indices);
	mesh.model = glm::translate(glm::mat4(1.0f), glm::vec3(-GRIDSIZE/2, -15.0, -GRIDSIZE/2));

	Shader defaultShader({"shader/default.vs", "shader/default.fs"}, {"in_Position", "in_Normal"});

	Tiny::view.pipeline = [&](){											//Setup Drawing Pipeline

		Tiny::view.target(color::white);								//Target Screen

		defaultShader.use();														//Prepare Shader
		defaultShader.uniform("model", mesh.model);			//Set Model Matrix
		defaultShader.uniform("vp", cam::vp);						//View Projection Matrix
		mesh.render(GL_TRIANGLES);													//Render Model with Lines

	};

	Tiny::loop([&](){ //Autorotate Camera
	//	cam::pan(0.1f);
	});

	Tiny::quit();

	return 0;
}
