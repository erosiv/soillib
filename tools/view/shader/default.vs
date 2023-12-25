#version 330

in vec3 in_Position;
in vec3 in_Normal;

uniform mat4 model;
uniform mat4 vp;

out vec4 ex_Color;
out vec3 ex_FragPos;

void main(void) {

	// Fragment Position in Model Space

	ex_FragPos = (model * vec4(in_Position, 1.0f)).xyz;
	gl_Position = vp * vec4(ex_FragPos, 1.0f);
	
	// Color Computation

	vec3 ex_Albedo = vec3(1);

	float diffuse = dot(normalize(in_Normal), normalize(vec3(1, 2, 1)));
	diffuse = clamp(diffuse, 0.1, 0.9);
	
	float light = 0.3 + 0.8*diffuse;
	ex_Color = vec4(vec3(light), 1.0);
	ex_Color = vec4(light*ex_Albedo, 1.0f);
}
