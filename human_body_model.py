import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import pyrr
from TextureLoader import load_texture
from myCube import Cube
from objLoader import ObjLoader
from pyrr.matrix44 import multiply
# import numpy as np

def readMultiObj(path):
    file = open(path)
    cnt = 0
    i_list = []
    obj_list = []
    lines = file.readlines()
    for l in lines:
        if l[0] == 'o':
            i_list.append(cnt)
        cnt = cnt + 1

    for i in range(1, len(i_list)):
        obj_list.append(lines[i_list[i-1]:i_list[i]])
    return obj_list

vertex_src = """
#version 330

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec2 a_texture;

uniform mat4 model;
uniform mat4 projection;
uniform mat4 view;

out vec3 v_color;
out vec2 v_texture;

void main()
{
    gl_Position = projection * view * model * vec4(a_position, 1.0);
    v_texture = a_texture;
}
"""

fragment_src = """
# version 330

in vec2 v_texture;

out vec4 out_color;

uniform sampler2D s_texture;

void main()
{
    out_color = vec4(100.0, 100.0, 0.0, 1.0);
}
"""


# glfw callback functions
def window_resize(window, width, height):
    glViewport(0, 0, width, height)
    projection = pyrr.matrix44.create_perspective_projection_matrix(45, width / height, 0.1, 100)
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)


# init GLFW library
if not glfw.init():
    raise Exception("glfw not initialized")

#create window
window = glfw.create_window(1280, 720, "HUMAN MODEL", None, None)

if not window:
    glfw.terminate()
    raise Exception("window not created")

# set window position
glfw.set_window_pos(window, 400, 200)

# set callback function for resizing the window
glfw.set_window_size_callback(window, window_resize)

# make the context current
glfw.make_context_current(window)

shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER), compileShader(fragment_src, GL_FRAGMENT_SHADER))

projection = pyrr.matrix44.create_perspective_projection_matrix(45, 1280 / 720, 0.1, 100)
view = pyrr.matrix44.create_look_at(pyrr.Vector3([-50, 10, 0]), pyrr.Vector3([0, 0, 0]), pyrr.Vector3([0, 1, 0]))

body_parts = []

k_size = 29

VAO = glGenVertexArrays(k_size)
VBO = glGenBuffers(k_size)
EBO = glGenBuffers(k_size)
indices = []
buffers = []
ignore = [3]

for i in range(0, k_size):
    human_indices, human_buffer = ObjLoader.load_model('model/' + str(i) + '.obj')
    indices.append(human_indices)
    buffers.append(human_buffer)
    if i in ignore:
        continue
    glBindVertexArray(VAO[i])
    glBindBuffer(GL_ARRAY_BUFFER, VBO[i])
    glBufferData(GL_ARRAY_BUFFER, human_buffer.nbytes, human_buffer, GL_STATIC_DRAW)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO[i])
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, human_buffer.nbytes, human_buffer, GL_STATIC_DRAW)

    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, human_buffer.itemsize * 8, ctypes.c_void_p(0))

    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, human_buffer.itemsize * 8, ctypes.c_void_p(12))

    glEnableVertexAttribArray(2)
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, human_buffer.itemsize * 8, ctypes.c_void_p(20))

# arms indices
# right lower: 15
# right upper: 14
# right joint upper: 13
# right joint lower: 12
# right hand: 11
# right shoulder: 10
# left lower: 4
# left upper: 5
# left joint upper: 6
# left joint lower: 7
# left hand: 8
# left shoulder: 9

# [shoulder, upper, joint_0, lower, joint_1, hand]
right_arm = [10, 14, 13, 15, 12, 11]
left_arm = [9, 5, 6, 4, 7, 8]


glUseProgram(shader)
glClearColor(0, 0.1, 0.1, 1)
glEnable(GL_DEPTH_TEST)
glEnable(GL_BLEND)
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

model_loc = glGetUniformLocation(shader, "model")
proj_loc = glGetUniformLocation(shader, "projection")
view_loc = glGetUniformLocation(shader, "view")

scale = pyrr.Matrix44.from_scale(pyrr.Vector3([0.01, 0.01, 0.01]))
glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)
glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)

while not glfw.window_should_close(window):
    glfw.poll_events()

    model = pyrr.Matrix44.identity()
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    t = glfw.get_time()
    rot_x = pyrr.matrix44.create_from_x_rotation(90)
    rot_y = pyrr.matrix44.create_from_y_rotation(t)
    rot_z = pyrr.matrix44.create_from_z_rotation(t)
    rot = multiply(rot_z, multiply(rot_x, rot_y))
    for i in range(0, k_size):
        glBindVertexArray(VAO[i])
        glUniformMatrix4fv(model_loc, 1, GL_FALSE, multiply(rot, scale))
        glDrawArrays(GL_TRIANGLES, 0, len(indices[i]))
    # glDrawElements(GL_TRIANGLES, len(human_indices), GL_UNSIGNED_INT, None)
    # angle = glfw.get_time()
    # HR.rotate([angle, 0, 0], [0, 0.5, 0])
    # for part in body_parts:
    #     part.update(model_loc)
    glfw.swap_buffers(window)