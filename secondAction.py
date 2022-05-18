import os

from face import Face
from vertex import Vertex
import math
import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
from OpenGL.raw.GL.VERSION.GL_1_5 import GL_STATIC_DRAW
import pyrr
from OpenGL.raw.GL.VERSION.GL_1_0 import GL_BLEND, GL_ONE_MINUS_SRC_ALPHA
from pyrr import Vector3, vector, vector3, matrix44
from math import sin, cos, radians
from objLoader import ObjLoader

os.environ['SDL_VIDEO_WINDOW_POS'] = '400,200'


class Camera:
    def __init__(self):
        self.camera_pos = Vector3([0.0, 4.0, 3.0])
        self.camera_front = Vector3([0.0, 0.0, -1.0])
        self.camera_up = Vector3([0.0, 1.0, 0.0])
        self.camera_right = Vector3([1.0, 0.0, 0.0])

        self.mouse_sensitivity = 0.25
        self.jaw = -90
        self.pitch = 0

    def get_view_matrix(self):
        return matrix44.create_look_at(self.camera_pos, self.camera_pos + self.camera_front, self.camera_up)

    def process_mouse_movement(self, xoffset, yoffset, constrain_pitch=True):
        xoffset *= self.mouse_sensitivity
        yoffset *= self.mouse_sensitivity

        self.jaw += xoffset
        self.pitch += yoffset

        if constrain_pitch:
            if self.pitch > 45:
                self.pitch = 45
            if self.pitch < -45:
                self.pitch = -45

        self.update_camera_vectors()

    def update_camera_vectors(self):
        front = Vector3([0.0, 0.0, 0.0])
        front.x = cos(radians(self.jaw)) * cos(radians(self.pitch))
        front.y = sin(radians(self.pitch))
        front.z = sin(radians(self.jaw)) * cos(radians(self.pitch))

        self.camera_front = vector.normalise(front)
        self.camera_right = vector.normalise(vector3.cross(self.camera_front, Vector3([0.0, 1.0, 0.0])))
        self.camera_up = vector.normalise(vector3.cross(self.camera_right, self.camera_front))

    # Camera method for the WASD movement
    def process_keyboard(self, direction, velocity):
        if direction == "FORWARD":
            self.camera_pos += self.camera_front * velocity
        if direction == "BACKWARD":
            self.camera_pos -= self.camera_front * velocity
        if direction == "LEFT":
            self.camera_pos -= self.camera_right * velocity
        if direction == "RIGHT":
            self.camera_pos += self.camera_right * velocity


cam = Camera()
WIDTH, HEIGHT = 1080, 680
lastX, lastY = WIDTH / 2, HEIGHT / 2
first_mouse = True
left, right, forward, backward = False, False, False, False


# the keyboard input callback
def key_input_clb(window, key, scancode, action, mode):
    global left, right, forward, backward
    if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
        glfw.set_window_should_close(window, True)

    if key == glfw.KEY_W and action == glfw.PRESS:
        forward = True
    elif key == glfw.KEY_W and action == glfw.RELEASE:
        forward = False
    if key == glfw.KEY_S and action == glfw.PRESS:
        backward = True
    elif key == glfw.KEY_S and action == glfw.RELEASE:
        backward = False
    if key == glfw.KEY_A and action == glfw.PRESS:
        left = True
    elif key == glfw.KEY_A and action == glfw.RELEASE:
        left = False
    if key == glfw.KEY_D and action == glfw.PRESS:
        right = True
    elif key == glfw.KEY_D and action == glfw.RELEASE:
        right = False
    # if key in [glfw.KEY_W, glfw.KEY_S, glfw.KEY_D, glfw.KEY_A] and action == glfw.RELEASE:
    #     left, right, forward, backward = False, False, False, False


# do the movement, call this function in the main loop
def do_movement():
    if left:
        cam.process_keyboard("LEFT", 0.05)
    if right:
        cam.process_keyboard("RIGHT", 0.05)
    if forward:
        cam.process_keyboard("FORWARD", 0.05)
    if backward:
        cam.process_keyboard("BACKWARD", 0.05)


# the mouse position callback function
def mouse_look_clb(window, xpos, ypos):
    global first_mouse, lastX, lastY

    if first_mouse:
        lastX = xpos
        lastY = ypos
        first_mouse = False

    xoffset = xpos - lastX
    yoffset = lastY - ypos

    lastX = xpos
    lastY = ypos

    cam.process_mouse_movement(xoffset, yoffset)


vertex_src = """
# version 330 core

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec2 a_texture;

uniform mat4 model;
uniform mat4 projection;
uniform mat4 view;

out vec3 v_color;
out vec2 v_texture;

void main()
{
    gl_Position = projection * view * model * vec4(a_position,1.0);
    v_texture = a_texture;
}
"""

fragment_src = """
# version 330 core

in vec2 v_texture;

out vec4 out_color;

uniform sampler2D s_texture;

void main()
{
    out_color = vec4(200.0, 0.0, 100.0, 1.0);
}
"""


def window_resize(window, width, height):
    glViewport(0, 0, width, height)
    projection = pyrr.matrix44.create_perspective_projection_matrix(45, width / height, 0.1, 100)
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)


def matrixMulpiple(matrixList):
    result = pyrr.matrix44.create_identity()
    for matrix in matrixList:
        result = pyrr.matrix44.multiply(result, matrix)
    return result


def draw(VAO, matrix, model_loc, indices):
    glBindVertexArray(VAO)
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, matrix)
    # glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)
    glDrawArrays(GL_TRIANGLES, 0, len(indices))


if not glfw.init():
    raise Exception("glfw init warning")

window = glfw.create_window(1280, 720, "openGl window", None, None)

if not window:
    glfw.terminate()
    raise Exception("window creating fail")

glfw.set_window_pos(window, 50, 50)
glfw.set_window_size_callback(window, window_resize)
# set the mouse position callback
glfw.set_cursor_pos_callback(window, mouse_look_clb)
# set the keyboard input callback
glfw.set_key_callback(window, key_input_clb)
# capture the mouse cursor
glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)

glfw.make_context_current(window)

# 两参数，vertex shaders—�?�它们可以处理顶点数据；以及fragment shaders，它们处理光栅化后生成的fragments�?
shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER), compileShader(fragment_src, GL_FRAGMENT_SHADER))

k_size = 30

VAO = glGenVertexArrays(k_size)
VBO = glGenBuffers(k_size)
EBO = glGenBuffers(k_size)
indices = []
buffers = []

for i in range(0, k_size):
    human_indices, human_buffer = ObjLoader.load_model('model/' + str(i) + '.obj')
    indices.append(human_indices)
    buffers.append(human_buffer)

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
right_arm = [10, 14, 13, 15, 12, 11, 40, 44, 43, 45, 42, 41]
left_arm = [9, 5, 6, 4, 7, 8, 39, 35, 36, 34, 37, 38]

glUseProgram(shader)
# RGB上限�?1
glClearColor(1, 1, 1, 1)

# 启动遮罩
glEnable(GL_DEPTH_TEST)
glEnable(GL_BLEND)
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

# param float fovy: field of view in y direction in degrees非零
# param float aspect: aspect ratio of the view (width / height)
# param float near: distance from the viewer to the near clipping plane (only positive)
# param float far: distance from the viewer to the far clipping plane (only positive)
projection = pyrr.matrix44.create_perspective_projection_matrix(45, 1080 / 860, 0.1, 100)

model_loc = glGetUniformLocation(shader, "model")
proj_loc = glGetUniformLocation(shader, "projection")
view_loc = glGetUniformLocation(shader, "view")

glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)
scale = pyrr.Matrix44.from_scale(pyrr.Vector3([0.01, 0.01, 0.01]))
rot_z = pyrr.matrix44.create_from_x_rotation(3.14 / 2)
model = pyrr.matrix44.multiply(rot_z, scale)
cj0 = pyrr.Vector4([0, 10.5, 0, 1.0])


def background2(_texture):
    try:
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_COLOR_MATERIAL)
        glBegin(GL_QUADS)
        glColor4f(0.25, 0.25, 0.25, 0.8)
        glVertex3d(-256, -256, 0)
        glColor4f(0.25, 0.25, 0.25, 0.8)
        glVertex3d(256, -256, 0)
        glColor4f(0.25, 0.25, 0.25, 0.8)
        glVertex3d(256, 256, 0)
        glColor4f(0.25, 0.25, 0.25, 0.8)
        glVertex3d(-256, 256, 0)
        glEnd()
        glFlush()
    except Exception as e:
        print(e)


def background():
    # glEnable(GL_TEXTURE_2D)
    # glBindTexture(GL_TEXTURE_2D, _texture)

    glBegin(GL_QUADS)
    # glTexCoord2d(0, 0)
    glVertex3d(-250, 0, 200)
    # glTexCoord2d(1, 0)
    glVertex3d(250, 0, 200)
    # glTexCoord2d(1, 1)
    glVertex3d(260, 500, 200)
    # glTexCoord2d(0, 1)
    glVertex3d(-250, 500, 200)
    glEnd()

    glBegin(GL_QUADS)
    glTexCoord2d(0, 0)
    glVertex3d(-250, 0, 700)
    glTexCoord2d(1, 0)
    glVertex3d(250, 0, 700)
    glTexCoord2d(1, 1)
    glVertex3d(250, 500, 700)
    glTexCoord2d(0, 1)
    glVertex3d(-250, 500, 700)
    glEnd()

    glBegin(GL_QUADS)
    glTexCoord2d(0, 0)
    glVertex3d(-250, 0, 200)
    glTexCoord2d(1, 0)
    glVertex3d(250, 0, 200)
    glTexCoord2d(1, 1)
    glVertex3d(250, 0, 700)
    glTexCoord2d(0, 1)
    glVertex3d(-250, 0, 700)
    glEnd()

    glBegin(GL_QUADS)
    glTexCoord2d(0, 0)
    glVertex3d(-250, 500, 200)
    glTexCoord2d(1, 0)
    glVertex3d(250, 500, 200)
    glTexCoord2d(1, 1)
    glVertex3d(250, 500, 700)
    glTexCoord2d(0, 1)
    glVertex3d(-250, 500, 700)
    glEnd()

    glBegin(GL_QUADS)
    glTexCoord2d(0, 0)
    glVertex3d(-250, 0, 200)
    glTexCoord2d(1, 0)
    glVertex3d(-250, 500, 200)
    glTexCoord2d(1, 1)
    glVertex3d(-250, 500, 700)
    glTexCoord2d(0, 1)
    glVertex3d(-250, 0, 700)
    glEnd()

    glBegin(GL_QUADS)
    glTexCoord2d(0, 0)
    glVertex3d(250, 0, 200)
    glTexCoord2d(1, 0)
    glVertex3d(250, 500, 200)
    glTexCoord2d(1, 1)
    glVertex3d(250, 500, 700)
    glTexCoord2d(0, 1)
    glVertex3d(250, 0, 700)
    glEnd()


def load_material(file):
    vertexes = []
    faces = []
    with open(file, mode='r', encoding='utf-8') as f:
        line = f.readline()
        while line:
            values = line.split()
            if values == [] or values[0] == "#" or values[0] == "vt":
                line = f.readline()
                continue
            if values[0] == "v":
                vertex = Vertex(float(values[1]), float(values[2]), float(values[3]))
                vertexes.append(vertex)
            elif values[0] == "f":
                face = Face(int(values[1]), int(values[2]), int(values[3]))
                faces.append(face)
            line = f.readline()
    for face in faces:
        glBegin(GL_TRIANGLES)
        glVertex3f(vertexes[face.v1 - 1].x, vertexes[face.v1 - 1].y, vertexes[face.v1 - 1].z)
        glVertex3f(vertexes[face.v2 - 1].x, vertexes[face.v2 - 1].y, vertexes[face.v2 - 1].z)
        glVertex3f(vertexes[face.v3 - 1].x, vertexes[face.v3 - 1].y, vertexes[face.v3 - 1].z)
        glEnd()


while not glfw.window_should_close(window):

    # 让鼠标可以用
    glfw.poll_events()
    do_movement()
    view = cam.get_view_matrix()

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    load_material('newModel/' + 'target' + '.obj')
    # load_material('model/' + 'mask1' + '.obj')
    # background()
    #     camX = math.sin(glfw.get_time()) * 20
    #     camZ = math.cos(glfw.get_time()) * 20
    #
    #     view = pyrr.matrix44.create_look_at(pyrr.Vector3([camX, 15.0, camZ]),
    #                                         pyrr.Vector3([0.0 , 0.0, 0.0]),
    #                                         pyrr.Vector3([0.0 , 1.0, 0.0]))
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)

    rot = 0

    t = (glfw.get_time() * 2) % 10.0

    fall = (t - 5) * (t - 5) * (t - 5) / 3
    if fall < 0:
        fall = 0
    if fall > 15:
        fall = 15
    if t < 1.57:
        rot = t * t * t / 1.57 / 1.57
    elif t < 4.71:
        rot = 1.57 + (t - 1.57) * (t - 1.57) * (t - 1.57) / 3.14 / 3.14 / 2
    elif t < 6.28:
        rot = 3.14
    elif t < 10:
        rot = 3.14  # - (t - 6.28) * (t - 6.28) * (t - 6.28) / 3.14 / 3.14

    for i in range(k_size):
        if i not in right_arm and i not in left_arm:
            draw(VAO[i], model, model_loc, indices[i])

    leftBigHandMatrix = matrixMulpiple([
        # pyrr.matrix44.create_from_scale(pyrr.Vector3([0.5, 5, 0.5])),
        model,
        pyrr.matrix44.create_from_translation(pyrr.Vector3([0, -13.5, 0])),
        pyrr.matrix44.create_from_axis_rotation(pyrr.Vector3([5.0, 1.0, 0]), -1 * rot),
        pyrr.matrix44.create_from_axis_rotation(pyrr.Vector3([0, 0, 5.0]), -0.1 * rot),
        pyrr.matrix44.create_from_translation(pyrr.Vector3([0, 13.5, 0])),
        # pyrr.matrix44.create_from_translation(pyrr.Vector3([-1.5, 3.5, 0]))
    ])
    for i in range(0, 3):
        draw(VAO[left_arm[i]], leftBigHandMatrix, model_loc, indices[left_arm[i]])
    # draw(VAO[left_arm[2]], leftBigHandMatrix, model_loc, indices[left_arm[2]])

    rightBigHandMatrix = matrixMulpiple([
        model,
        # pyrr.matrix44.create_from_scale(pyrr.Vector3([0.5, 5, 0.5])),
        pyrr.matrix44.create_from_translation(pyrr.Vector3([0, -13.5, 0])),
        pyrr.matrix44.create_from_axis_rotation(pyrr.Vector3([5.0, -1.0, 0]), -1 * rot),
        pyrr.matrix44.create_from_axis_rotation(pyrr.Vector3([0, 0, 5.0]), 0.1 * rot),
        pyrr.matrix44.create_from_translation(pyrr.Vector3([0, 13.5, 0])),
        # pyrr.matrix44.create_from_translation(pyrr.Vector3([1.5, 3.5, 0]))
    ])
    for i in range(0, 3):
        draw(VAO[right_arm[i]], rightBigHandMatrix, model_loc, indices[right_arm[i]])

    result = matrixMulpiple([pyrr.matrix44.create_from_translation(pyrr.Vector3([0, -13.5, 0])),
                             pyrr.matrix44.create_from_axis_rotation(pyrr.Vector3([5.0, 0, 0]), -0.98 * rot),
                             pyrr.matrix44.create_from_translation(pyrr.Vector3([0, 13.5, 0])), ])

    center = pyrr.matrix44.apply_to_vector(result, cj0)

    record = pyrr.matrix44.apply_to_vector(
        matrixMulpiple([pyrr.matrix44.create_from_translation(pyrr.Vector3([0, -13.5, 0])),
                        pyrr.matrix44.create_from_axis_rotation(pyrr.Vector3([5.0, 0, 0]), -0.98 * 1.57),
                        pyrr.matrix44.create_from_translation(pyrr.Vector3([0, 13.5, 0])), ]), cj0)

    leftSmallHandMatrix = matrixMulpiple([
        # pyrr.matrix44.create_from_scale(pyrr.Vector3([0.5, 5, 0.5])),
        model,
        pyrr.matrix44.create_from_translation(pyrr.Vector3([0, -13.5, 0])),
        pyrr.matrix44.create_from_axis_rotation(pyrr.Vector3([0.75, 0, 0]), -0.8 * rot),
        pyrr.matrix44.create_from_axis_rotation(pyrr.Vector3([0, 0.75, 0]), -0.1 * rot),
        pyrr.matrix44.create_from_translation(pyrr.Vector3([0, 13.5, 0])),
        pyrr.matrix44.create_from_translation(-center),
        pyrr.matrix44.create_from_axis_rotation(pyrr.Vector3([5.0, 0, 0]), -1 * rot),
        pyrr.matrix44.create_from_axis_rotation(pyrr.Vector3([0, 0, 5.0]), -0.1 * rot),
        pyrr.matrix44.create_from_translation(center)
    ])

    armLength = 3
    if 1.57 < t < 4.71:
        leftSmallHandMatrix = matrixMulpiple([
            model,
            pyrr.matrix44.create_from_translation(pyrr.Vector3([0, -13.5, 0])),
            pyrr.matrix44.create_from_axis_rotation(pyrr.Vector3([0.75, 0, 0]), -0.8 * 1.57),
            pyrr.matrix44.create_from_axis_rotation(pyrr.Vector3([0, 0.75, 0]), -0.1 * 1.57),
            pyrr.matrix44.create_from_translation(pyrr.Vector3([0, 13.5, 0])),
            pyrr.matrix44.create_from_translation(-record),
            pyrr.matrix44.create_from_axis_rotation(pyrr.Vector3([5.0, 0, 0]), -1 * 1.57),
            pyrr.matrix44.create_from_axis_rotation(pyrr.Vector3([0, 0, 5.0]), -0.1 * 1.57),
            pyrr.matrix44.create_from_translation(record),
            pyrr.matrix44.create_from_translation(pyrr.Vector3([0, (armLength - 0.3) * math.sin(rot - 1.57), 0])),  # 向上
            pyrr.matrix44.create_from_translation(pyrr.Vector3([0, 0, (armLength + 0.5) * (math.cos(rot - 1.57) - 1)]))
            # 向后
        ])
    if 4.71 <= t < 10:
        leftSmallHandMatrix = matrixMulpiple([
            model,
            pyrr.matrix44.create_from_translation(pyrr.Vector3([0, -13.5, 0])),
            pyrr.matrix44.create_from_axis_rotation(pyrr.Vector3([0.75, 0, 0]), -0.8 * 1.57),
            pyrr.matrix44.create_from_axis_rotation(pyrr.Vector3([0, 0.75, 0]), -0.1 * 1.57),
            pyrr.matrix44.create_from_translation(pyrr.Vector3([0, 13.5, 0])),
            pyrr.matrix44.create_from_translation(-record),
            pyrr.matrix44.create_from_axis_rotation(pyrr.Vector3([5.0, 0, 0]), -1 * 1.57),
            pyrr.matrix44.create_from_axis_rotation(pyrr.Vector3([0, 0, 5.0]), -0.1 * 1.57),
            pyrr.matrix44.create_from_translation(record),
            pyrr.matrix44.create_from_translation(pyrr.Vector3([0, (armLength - 0.3), 0])),
            pyrr.matrix44.create_from_translation(pyrr.Vector3([0, 0, -(armLength + 0.5)]))
        ])
    for i in range(3, 6):
        draw(VAO[left_arm[i]], leftSmallHandMatrix, model_loc, indices[left_arm[i]])

    rightSmallHandMatrix = matrixMulpiple([
        # pyrr.matrix44.create_from_scale(pyrr.Vector3([0.5, 5, 0.5])),
        model,
        pyrr.matrix44.create_from_translation(pyrr.Vector3([0, -13.5, 0])),
        pyrr.matrix44.create_from_axis_rotation(pyrr.Vector3([0.75, 0, 0]), -0.8 * rot),
        pyrr.matrix44.create_from_axis_rotation(pyrr.Vector3([0, 0.75, 0]), -0.1 * rot),
        pyrr.matrix44.create_from_translation(pyrr.Vector3([0, 13.5, 0])),
        pyrr.matrix44.create_from_translation(-center),
        pyrr.matrix44.create_from_axis_rotation(pyrr.Vector3([5.0, 0, 0]), -1 * rot),
        pyrr.matrix44.create_from_axis_rotation(pyrr.Vector3([0, 0, 5.0]), -0.1 * rot),
        pyrr.matrix44.create_from_translation(center),
        pyrr.matrix44.create_from_translation(pyrr.Vector3([0.3 * t, 0, 0]))
    ])

    if 1.57 < t < 4.71:
        rightSmallHandMatrix = matrixMulpiple([
            model,
            pyrr.matrix44.create_from_translation(pyrr.Vector3([0, -13.5, 0])),
            pyrr.matrix44.create_from_axis_rotation(pyrr.Vector3([0.75, 0, 0]), -0.8 * 1.57),
            pyrr.matrix44.create_from_axis_rotation(pyrr.Vector3([0, 0.75, 0]), -0.1 * 1.57),
            pyrr.matrix44.create_from_translation(pyrr.Vector3([0, 13.5, 0])),
            pyrr.matrix44.create_from_translation(-record),
            pyrr.matrix44.create_from_axis_rotation(pyrr.Vector3([5.0, 0, 0]), -1 * 1.57),
            pyrr.matrix44.create_from_axis_rotation(pyrr.Vector3([0, 0, 5.0]), -0.1 * 1.57),
            pyrr.matrix44.create_from_translation(record),
            pyrr.matrix44.create_from_translation(pyrr.Vector3([0, (armLength - 0.3) * math.sin(rot - 1.57), 0])),  # 向上
            pyrr.matrix44.create_from_translation(pyrr.Vector3([0, 0, (armLength + 0.5) * (math.cos(rot - 1.57) - 1)])),  # 向后
            pyrr.matrix44.create_from_translation(pyrr.Vector3([0.471+0.15 * (t-1.57), 0, 0]))

        ])
    if 4.71 <= t < 10:
        rightSmallHandMatrix = matrixMulpiple([
            model,
            pyrr.matrix44.create_from_translation(pyrr.Vector3([0, -13.5, 0])),
            pyrr.matrix44.create_from_axis_rotation(pyrr.Vector3([0.75, 0, 0]), -0.8 * 1.57),
            pyrr.matrix44.create_from_axis_rotation(pyrr.Vector3([0, 0.75, 0]), -0.1 * 1.57),
            pyrr.matrix44.create_from_translation(pyrr.Vector3([0, 13.5, 0])),
            pyrr.matrix44.create_from_translation(-record),
            pyrr.matrix44.create_from_axis_rotation(pyrr.Vector3([5.0, 0, 0]), -1 * 1.57),
            pyrr.matrix44.create_from_axis_rotation(pyrr.Vector3([0, 0, 5.0]), -0.1 * 1.57),
            pyrr.matrix44.create_from_translation(record),
            pyrr.matrix44.create_from_translation(pyrr.Vector3([0, (armLength - 0.3), 0])),
            pyrr.matrix44.create_from_translation(pyrr.Vector3([0, 0, -(armLength + 0.5)])),
            pyrr.matrix44.create_from_translation(pyrr.Vector3([0.942, 0, 0]))
        ])
    for i in range(3, 6):
        draw(VAO[right_arm[i]], rightSmallHandMatrix, model_loc, indices[right_arm[i]])
    # draw(VAO[right_arm[3]], rightSmallHandMatrix, model_loc, indices[right_arm[3]])
    # draw(VAO[right_arm[3]], rightSmallHandMatrix, model_loc, indices[right_arm[3]])
    # BallMatrix = matrixMulpiple([
    #     # pyrr.matrix44.create_from_scale(pyrr.Vector3([1,1,1])),
    #     scale,
    #     pyrr.matrix44.create_from_translation(pyrr.Vector3([0, 7.5-fall*0.7, fall/1.2]))
    # ])
    # draw(VAO, BallMatrix, model_loc, indices)

    leftBigHandMatrix = matrixMulpiple([
        # pyrr.matrix44.create_from_scale(pyrr.Vector3([0.5, 5, 0.5])),
        model,
        pyrr.matrix44.create_from_translation(pyrr.Vector3([0, -13.5, 0])),
        pyrr.matrix44.create_from_axis_rotation(pyrr.Vector3([5.0, 1.0, 0]), -1 * rot),
        pyrr.matrix44.create_from_axis_rotation(pyrr.Vector3([0, 0, 5.0]), -0.1 * rot),
        pyrr.matrix44.create_from_translation(pyrr.Vector3([0, 13.5, 0])),
        # pyrr.matrix44.create_from_translation(pyrr.Vector3([-1.5, 3.5, 0]))
    ])

    '''for i in range(6, 9):
        draw(VAO[left_arm[i]], leftBigHandMatrix, model_loc, indices[left_arm[i]])
    # draw(VAO[left_arm[2]], leftBigHandMatrix, model_loc, indices[left_arm[2]])

    rightBigHandMatrix = matrixMulpiple([
        # pyrr.matrix44.create_from_scale(pyrr.Vector3([0.5, 5, 0.5])),
        pyrr.matrix44.create_from_translation(pyrr.Vector3([0, -13.5, 0])),
        pyrr.matrix44.create_from_axis_rotation(pyrr.Vector3([5.0, -1.0, 0]), -1 * rot),
        pyrr.matrix44.create_from_axis_rotation(pyrr.Vector3([0, 0, 5.0]), 0.1 * rot),
        pyrr.matrix44.create_from_translation(pyrr.Vector3([0, 13.5, 0])),
        # pyrr.matrix44.create_from_translation(pyrr.Vector3([1.5, 3.5, 0]))
    ])
    for i in range(6, 9):
        draw(VAO[right_arm[i]], pyrr.matrix44.multiply(model, rightBigHandMatrix), model_loc, indices[right_arm[i]])
    # draw(VAO[right_arm[2]], pyrr.matrix44.multiply(rightBigHandMatrix, model), model_loc, indices[right_arm[2]])
    result = matrixMulpiple([pyrr.matrix44.create_from_translation(pyrr.Vector3([0, -13.5, 0])),
                             pyrr.matrix44.create_from_axis_rotation(pyrr.Vector3([5.0, 0, 0]), -0.98 * rot),
                             pyrr.matrix44.create_from_translation(pyrr.Vector3([0, 13.5, 0])), ])
    center = pyrr.matrix44.apply_to_vector(result, cj0)

    leftSmallHandMatrix = matrixMulpiple([
        # pyrr.matrix44.create_from_scale(pyrr.Vector3([0.5, 5, 0.5])),
        model,
        pyrr.matrix44.create_from_translation(pyrr.Vector3([0, -13.5, 0])),
        pyrr.matrix44.create_from_axis_rotation(pyrr.Vector3([0.75, 0, 0]), -0.8 * rot),
        pyrr.matrix44.create_from_axis_rotation(pyrr.Vector3([0, 0.75, 0]), -0.1 * rot),
        pyrr.matrix44.create_from_translation(pyrr.Vector3([0, 13.5, 0])),
        pyrr.matrix44.create_from_translation(-center),
        pyrr.matrix44.create_from_axis_rotation(pyrr.Vector3([5.0, 0, 0]), -1 * rot),
        pyrr.matrix44.create_from_axis_rotation(pyrr.Vector3([0, 0, 5.0]), -0.1 * rot),
        pyrr.matrix44.create_from_translation(center)
    ])
    for i in range(9, 12):
        draw(VAO[left_arm[i]], leftSmallHandMatrix, model_loc, indices[left_arm[i]])

    rightSmallHandMatrix = matrixMulpiple([
        # pyrr.matrix44.create_from_scale(pyrr.Vector3([0.5, 5, 0.5])),
        model,
        pyrr.matrix44.create_from_translation(pyrr.Vector3([0, -13.5, 0])),
        pyrr.matrix44.create_from_axis_rotation(pyrr.Vector3([0.75, 0, 0]), -0.8 * rot),
        pyrr.matrix44.create_from_axis_rotation(pyrr.Vector3([0, 0.75, 0]), 0.1 * rot),
        pyrr.matrix44.create_from_translation(pyrr.Vector3([0, 13.5, 0])),
        pyrr.matrix44.create_from_translation(-center),
        pyrr.matrix44.create_from_axis_rotation(pyrr.Vector3([5.0, 0, 0]), -1 * rot),
        pyrr.matrix44.create_from_axis_rotation(pyrr.Vector3([0, 0, 5.0]), 0.1 * rot),
        pyrr.matrix44.create_from_translation(center)
    ])
    for i in range(9, 12):
        draw(VAO[right_arm[i]], rightSmallHandMatrix, model_loc, indices[right_arm[i]])
    # draw(VAO[right_arm[3]], rightSmallHandMatrix, model_loc, indices[right_arm[3]])
    # draw(VAO[right_arm[3]], rightSmallHandMatrix, model_loc, indices[right_arm[3]])
    # BallMatrix = matrixMulpiple([
    #     # pyrr.matrix44.create_from_scale(pyrr.Vector3([1,1,1])),
    #     scale,
    #     pyrr.matrix44.create_from_translation(pyrr.Vector3([0, 7.5-fall*0.7, fall/1.2]))
    # ])
    # draw(VAO, BallMatrix, model_loc, indices)'''

    glfw.swap_buffers(window)

glfw.terminate()
