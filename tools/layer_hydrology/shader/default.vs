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

	// Compute Position

	ex_FragPos = (model * vec4(in_Position, 1.0f)).xyz;
	gl_Position = vp * vec4(ex_FragPos, 1.0f);

	//Fragment Position in Model Space

	vec3 normal = texture(normalMap, in_Position.xz/dimension).xyz;
	normal = normalize((normal - 0.5f)/0.5f);

	vec3 snormal =	texture(subNormalMap, in_Position.xz/dimension).xyz;
	snormal = normalize((snormal - 0.5f)/0.5f);

	float ex_Albedo = texture(albedoMap, in_Position.xz/dimension).a;
	float discharge = texture(dischargeMap, in_Position.xz/dimension).a;

	// Compute Color

	ex_Color = vec4(0.40, 0.60, 0.25, 1.0f);

	vec3 watercolor = vec3(0.27, 0.50, 0.54);
	vec3 steepcolor = vec3(0.7);

	if(normal.y < 0.8)
		ex_Color = vec4(steepcolor, 1.0f);

	vec3 lightpos = normalize(vec3(1, 1, 1));

	if(gl_VertexID / int((dimension.x-1)*(dimension.y-1)*6) == 0){

		float diffuse = clamp(dot(snormal, normalize(lightpos)), 0.0, 1.0);	
		float light = diffuse;

	//	if(ex_Albedo != 1)
	//		ex_Color = mix(ex_Color, vec4(0.1,0.1,0.1,1), 0.6*discharge);
	//	else
		ex_Color = vec4(watercolor,1.0);
		ex_Color = vec4(light*ex_Color.xyz, 1.0f);
	
	} else {

		float diffuse = clamp(dot(normal, normalize(lightpos)), 0.0, 1.0);	
		float light = 0.2 + diffuse;

		if(ex_Albedo == 0){
			ex_Color = mix(ex_Color, vec4(watercolor,1), discharge);
			ex_Color = vec4(light*ex_Color.xyz, 1.0);
		}
		else{

			ex_Color = mix(ex_Color, vec4(0.25,0.35,0.6,1), ex_Albedo);
			ex_Color = vec4(light*ex_Color.xyz, 0.5);
			ex_Color = mix(ex_Color, vec4(vec3(0.75,0.85,0.9),0.5), 0.2*discharge);

		}


	
	}


}
