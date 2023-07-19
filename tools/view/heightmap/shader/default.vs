#version 330

in vec3 in_Position;
in vec3 in_Normal;

uniform mat4 model;
uniform mat4 vp;

uniform sampler2D dischargeMap;
uniform sampler2D normalMap;

out vec4 ex_Color;
out vec3 ex_Normal;
out vec3 ex_FragPos;

void main(void) {
	//Fragment Position in Model Space
	ex_FragPos = (model * vec4(in_Position, 1.0f)).xyz;
	ex_Normal = texture(normalMap, in_Position.xz/vec2(512)).xyz;

	//Fragment in Screen Space
	gl_Position = vp * vec4(ex_FragPos, 1.0f);

	float discharge = texture(dischargeMap, in_Position.xz/vec2(512)).a;

	//Color from Normal Vector
	float light = dot(normalize(in_Normal), normalize(vec3(1, 1, 1)));
	ex_Color = vec4(vec3(light), 1.0);
	ex_Color = vec4(ex_Normal, 1.0f);
	ex_Color = mix(ex_Color, vec4(1,1,1,1), discharge);
}