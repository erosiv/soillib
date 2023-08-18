#version 330

in vec3 in_Position;

uniform mat4 model;
uniform mat4 vp;

uniform sampler2D dischargeMap;
uniform sampler2D normalMap;
uniform sampler2D subNormalMap;
uniform sampler2D albedoMap;
uniform vec2 dimension;

out vec4 ex_Color;
out vec3 ex_Normal;
out vec3 ex_FragPos;

uniform bool albedoRead;

void main(void) {
	//Fragment Position in Model Space
	ex_FragPos = (model * vec4(in_Position, 1.0f)).xyz;
	vec3 normal = texture(normalMap, in_Position.xz/dimension).xyz;
	vec3 snormal = texture(subNormalMap, in_Position.xz/dimension).xyz;
	float ex_Albedo = texture(albedoMap, in_Position.xz/dimension).a;

	//Fragment in Screen Space
	gl_Position = vp * vec4(ex_FragPos, 1.0f);
	ex_Color = vec4(1.0, 1.0, 1.0, 1.0f);

	float discharge = texture(dischargeMap, in_Position.xz/dimension).a;


	//vec3 subnormal = normalize(ex_Normal).zxy;



	if(gl_VertexID / int(dimension.x*dimension.y*6) == 0){

		float diffuse = clamp(dot(normalize(snormal), normalize(vec3(1, 1, 1))), 0.0, 1.0);	
		float light = diffuse;

		if(ex_Albedo != 1)
			ex_Color = mix(ex_Color, vec4(0.1,0.1,0.1,1), 0.6*discharge);

		ex_Color = vec4(light*ex_Color.xyz, 1.0f);
	
	} else {

		float diffuse = clamp(dot(normalize(normal), normalize(vec3(1, 1, 1))), 0.0, 1.0);	
		float light = 0.2 + 0.8*diffuse;

		if(ex_Albedo == 0){
			ex_Color = mix(ex_Color, vec4(0.1,0.1,0.1,1), 0.6*discharge);
			ex_Color = vec4(light*ex_Color.xyz, 1.0);
		}
		else{

			ex_Color = mix(ex_Color, vec4(0.25,0.35,0.6,1), ex_Albedo);
			ex_Color = vec4(light*ex_Color.xyz, 0.5);
			ex_Color = mix(ex_Color, vec4(0.75,0.85,0.9,1), discharge);

		}


	
	}


}
