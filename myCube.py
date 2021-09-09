import numpy as np
import pyrr
from OpenGL.GL import *

from TextureLoader import load_texture


class Cube:
    def __init__(self, width=1, height=1, length=1, pos=None):
        if pos is None:
            pos = [0.0, 0.0, 0.0]
        self.pos = np.array(pos)
        self.length = length
        self.width = width
        self.height = height
        self.vertices = np.array([
            -0.5, -0.5, 0.5, 0.0, 0.0,
            0.5, -0.5, 0.5, 1.0, 0.0,
            0.5, 0.5, 0.5, 1.0, 1.0,
            -0.5, 0.5, 0.5, 0.0, 1.0,

            -0.5, -0.5, -0.5, 0.0, 0.0,
            0.5, -0.5, -0.5, 1.0, 0.0,
            0.5, 0.5, -0.5, 1.0, 1.0,
            -0.5, 0.5, -0.5, 0.0, 1.0,

            0.5, -0.5, -0.5, 0.0, 0.0,
            0.5, 0.5, -0.5, 1.0, 0.0,
            0.5, 0.5, 0.5, 1.0, 1.0,
            0.5, -0.5, 0.5, 0.0, 1.0,

            -0.5, 0.5, -0.5, 0.0, 0.0,
            -0.5, -0.5, -0.5, 1.0, 0.0,
            -0.5, -0.5, 0.5, 1.0, 1.0,
            -0.5, 0.5, 0.5, 0.0, 1.0,

            -0.5, -0.5, -0.5, 0.0, 0.0,
            0.5, -0.5, -0.5, 1.0, 0.0,
            0.5, -0.5, 0.5, 1.0, 1.0,
            -0.5, -0.5, 0.5, 0.0, 1.0,

            0.5, 0.5, -0.5, 0.0, 0.0,
            -0.5, 0.5, -0.5, 1.0, 0.0,
            -0.5, 0.5, 0.5, 1.0, 1.0,
            0.5, 0.5, 0.5, 0.0, 1.0
        ], dtype=np.float32)
        self.indices = np.array([
            0, 1, 2, 2, 3, 0,
            4, 5, 6, 6, 7, 4,
            8, 9, 10, 10, 11, 8,
            12, 13, 14, 14, 15, 12,
            16, 17, 18, 18, 19, 16,
            20, 21, 22, 22, 23, 20
        ], dtype=np.uint32)
        self.pos_m = pyrr.Matrix44.from_translation(pyrr.Vector3(self.pos + 0.5))
        self.scale = pyrr.Matrix44.from_scale(pyrr.Vector3([self.width, self.height, self.length]))
        self.rot = None

    def genGlfwCube(self, texture_path):
        self.VAO = glGenVertexArrays(1)
        VBO = glGenBuffers(1)
        EBO = glGenBuffers(1)
        glBindVertexArray(self.VAO)
        glBindBuffer(GL_ARRAY_BUFFER, VBO)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_DRAW)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, self.vertices.itemsize * 5, ctypes.c_void_p(0))

        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, self.vertices.itemsize * 5, ctypes.c_void_p(12))

        self.texture = glGenTextures(1)

        load_texture(texture_path, self.texture)
        return

    def update(self, model_loc):
        glBindVertexArray(self.VAO)
        glBindTexture(GL_TEXTURE_2D, self.texture)

        move_back = pyrr.Matrix44.from_translation(pyrr.Vector3([-0.5, -0.5, -0.5]))
        move = pyrr.Matrix44.from_translation(pyrr.Vector3([0.5, 0.5, 0.5]))
        if self.rot is not None:
            model = self.rot * self.pos_m * move_back * self.scale * move
        else:
            model = self.pos_m * move_back * self.scale * move
        glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
        glDrawElements(GL_TRIANGLES, len(self.indices), GL_UNSIGNED_INT, None)

    def rotate(self, angle=None, ankle=None):
        if angle is None:
            self.rot = pyrr.Matrix44.identity()
            return
        if ankle is None:
            ankle = [0, 0, 0]
        move = pyrr.Matrix44.from_translation(-pyrr.Vector3(ankle))
        move_back = pyrr.Matrix44.from_translation(pyrr.Vector3(ankle))
        rot_x = pyrr.Matrix44.from_x_rotation(angle[0])
        rot_y = pyrr.Matrix44.from_y_rotation(angle[1])
        rot_z = pyrr.Matrix44.from_z_rotation(angle[2])
        self.rot = move_back * rot_x * rot_y * rot_z * move
