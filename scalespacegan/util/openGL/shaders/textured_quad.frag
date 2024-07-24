#version 420

layout (binding = 0) uniform sampler2D inputTexture2D;
layout (binding = 1) uniform sampler2DArray inputTexture2DArray;
layout (binding = 2) uniform sampler2D overlayTexture2D;

uniform ivec2 outputRes;
uniform bool showArray = false;
uniform int level = 0;
uniform int layer = 0;
uniform bool showOverlay = false;
uniform bool alpha_composite_overlay = true;
uniform ivec2 overlayPosition = ivec2(0);

in vec2 uv;

out vec4 out_color;


void main()
{	
	ivec2 inputRes = (!showArray) ? textureSize(inputTexture2D, 0) : textureSize(inputTexture2DArray, 0).xy;
	
	vec2 customUV = uv;
	if (outputRes != inputRes)
	{
		vec2 ratios = vec2(outputRes) / inputRes;
		customUV *= ratios / min(ratios.x, ratios.y);
	}
	if (!showArray) out_color = textureLod(inputTexture2D, customUV, level);
	else out_color = textureLod(inputTexture2DArray, vec3(customUV, layer), level);

	if (showOverlay)
	{
		ivec2 fetchCoord = ivec2(gl_FragCoord.xy) - overlayPosition;
		ivec2 overlayRes = textureSize(overlayTexture2D, 0);
		if (all(lessThan(fetchCoord, overlayRes)) && all(greaterThanEqual(fetchCoord, ivec2(0))))
		{
			vec4 overlay = texelFetch(overlayTexture2D, fetchCoord, 0);
			// out_color.rgb = overlay.rgb + (1 - overlay.a) * out_color.rgb;
			if (alpha_composite_overlay){
				out_color.rgb = overlay.rgb + (1 - overlay.a) * out_color.rgb;
				
			}
			else{
				if (any(greaterThan(overlay.rgb, vec3(0.2)))){
					out_color.rgb = overlay.rgb;
				}
				else{
					out_color.rgb = (out_color.rgb);
				}
				
			}
			
		}
	}
}