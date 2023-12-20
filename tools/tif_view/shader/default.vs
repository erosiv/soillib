#version 330

in vec3 in_Position;

uniform mat4 model;
uniform mat4 vp;

uniform sampler2D normalMap;
uniform vec2 dimension;

out vec4 ex_Color;
out vec3 ex_Normal;
out vec3 ex_FragPos;

void main(void) {
	//Fragment Position in Model Space
	ex_FragPos = (model * vec4(in_Position, 1.0f)).xyz;
	ex_Normal = texture(normalMap, in_Position.xz/dimension).xyz;

	//Fragment in Screen Space
	gl_Position = vp * vec4(ex_FragPos, 1.0f);


	vec3 normal = normalize(ex_Normal);

	float diffuse = dot(normalize(normal), normalize(vec3(-1, 4, -1)));
	diffuse = clamp(diffuse, 0.1, 0.9);
	
	float light = 0.3 + 0.8*diffuse;

	ex_Normal = 0.5+0.5*ex_Normal;
	//ex_Normal.y = 1.0-ex_Normal.y;
//	ex_Color = vec4(vec3(light), 1.0);
	ex_Color = vec4(light*vec3(1.0), 1.0f);
	ex_Color = vec4(ex_Normal, 1.0f);
}
