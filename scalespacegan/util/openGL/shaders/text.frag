#version 460

layout (binding=0) uniform sampler2D u_texture;

in vec2 uv;

uniform vec3 textColor;

out vec4 fragColor;

void main()
{
    float text = texture(u_texture, uv).r;
    fragColor = vec4(textColor * text, text);
}