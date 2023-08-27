#version 330 core

in vec3 ex_Normal;
out vec4 ex_Color;

uniform vec3 color;

void main(){

  vec3 lightpos = normalize(vec3(1, 1, 1));

  float diffuse = clamp(dot(normalize(ex_Normal), normalize(lightpos)), 0.1, 0.9); 
  float light = 0.6 + diffuse;

  ex_Color = vec4(light*color, 1);

}
