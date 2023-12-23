#version 330

in vec3 in_Position;

uniform mat4 model;
uniform mat4 vp;

//uniform sampler2D dischargeMap;
uniform sampler2D normalMap;
//uniform sampler2D albedoMap;
uniform vec2 dimension;

out vec4 ex_Color;
out vec3 ex_Normal;
out vec3 ex_FragPos;

void main(void) {

	//Fragment Position in Model Space
	ex_FragPos = (model * vec4(in_Position, 1.0f)).xyz;
	gl_Position = vp * vec4(ex_FragPos, 1.0f); // Screen Space

	vec3 ex_Normal = texture(normalMap, in_Position.xz/dimension).xyz;
	float diffuse = dot(normalize(ex_Normal), normalize(vec3(1, 2, 1)));
	diffuse = clamp(diffuse, 0.05, 0.95);

	vec3 ex_Albedo = vec3(1);
	float light = 0.1 + 0.9*diffuse;
	ex_Color = vec4(light*ex_Albedo, 1.0f);

	if(ex_Normal.y > 0.9975)
		ex_Color = vec4(92, 133, 142, 255.0f)/255.0f;
	
	/*
	vec3 normal = ex_Normal;
	normal = 0.5*normal + 0.5;
	normal.yz = normal.zy;
	ex_Color = vec4(normal, 1.0f);
	*/
	
}