#version 420 core

layout(location = 0) in vec2 position;

uniform mat4 modelMatrix = mat4(1);

out vec2 uv;

void main()
{
    gl_Position = modelMatrix * vec4(position, 0, 1);
    uv = vec2(position.x, 1 + position.y);
}