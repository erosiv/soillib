#version 330

in vec3 in_Position;

uniform mat4 model;
uniform mat4 vp;

uniform sampler2D dischargeMap;
uniform sampler2D normalMap;
uniform sampler2D subNormalMap;
uniform sampler2D albedoMap;
uniform sampler2D hdiffMap;
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

	float hdiff = texture(hdiffMap, in_Position.xz/dimension).a;

	// Compute Color

	ex_Color = vec4(1.0f);

	vec3 watercolor = vec3(0.27, 0.50, 0.54);
	vec3 lightpos = normalize(vec3(1, 1, 1));



	// Sub-Surface Coloring

	if(gl_VertexID / int((dimension.x-1)*(dimension.y-1)*6) != 1){

		float diffuse = clamp(dot(snormal, normalize(lightpos)), 0.1, 0.9);	
		float light = 0.2 + diffuse;

		ex_Color = mix(vec4(ex_Color), vec4(watercolor, 1.0), discharge);//;mix(ex_Color,, ex_Albedo);
		ex_Color = mix(ex_Color, vec4(watercolor, 1.0), 1.0f - exp(-100*hdiff));
		ex_Color = mix(ex_Color, vec4(0.1, 0.1, 0.1, 1.0), 1.0f - exp(-10*hdiff));
		ex_Color = vec4(light*ex_Color.xyz, 1.0f);

		return;

	} 

	// Surface Coloring

	float diffuse = clamp(dot(normal, normalize(lightpos)), 0.1, 0.9);	
	float light = 0.2 + diffuse;

	// Regular Land

	if(ex_Albedo != 1){

		ex_Color = mix(ex_Color, vec4(watercolor,1), discharge);
		ex_Color = vec4(light*ex_Color.xyz, 1.0);

	}
	else{

		float depth = 1.0f-exp(-10*hdiff);


		ex_Color.xyz = watercolor;
		ex_Color.xyz = mix(ex_Color.xyz, watercolor, depth);
		ex_Color.a = depth;
	//	ex_Color = mix(ex_Color, vec4(watercolor, 1.0f), discharge);
//		ex_Color = vec4(watercolor, 0.5);

	}

}
