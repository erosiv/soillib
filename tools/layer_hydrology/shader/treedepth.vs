#version 330 core

in vec4 in_Pos;
in vec3 in_Normal;
in mat4 in_Model;

uniform mat4 dvp;

void main(){

  gl_Position = dvp*in_Model*in_Pos;

}
