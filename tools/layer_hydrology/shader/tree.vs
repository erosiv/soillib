#version 330 core

in vec4 in_Pos;
in vec3 in_Normal;
in mat4 in_Model;

out vec3 ex_Normal;

uniform mat4 proj;
uniform mat4 view;

void main(void) {

	vec4 v_Position = view * in_Model * in_Pos;
	gl_Position = proj*v_Position;
	ex_Normal = in_Normal;

}
