#version 420 core

layout(location = 0) in vec2 position;

uniform mat4 modelMatrix = mat4(1);

out vec2 uv;
out vec2 vLineCenter;

void main()
{
    gl_Position = modelMatrix * vec4(position, 0, 1);
    uv = 0.5 * (position.xy + vec2(1));
    vec2 vLineCenter = 0.5*(gl_Position.xy + vec2(1, 1));
}