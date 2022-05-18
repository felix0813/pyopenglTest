import time

import numpy as np
from OpenGL.GL import *
import math
from face import Face
from vertex import Vertex


class ObjLoader:
    buffer = []

    @staticmethod
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
                    vertex = Vertex(values[1], values[2], values[3])
                    vertexes.append(vertex)
                elif values[0] == "f":
                    face = Face(values[1].split('/')[0], values[2].split('/')[0], values[3].split('/')[0])
                    faces.append(face)
                line = f.readline()
        for face in faces:
            glBegin(GL_TRIANGLES)
            glVertex3f(vertexes[face.v1 - 1].x, vertexes[face.v1 - 1].y, vertexes[face.v1 - 1].z)
            glVertex3f(vertexes[face.v2 - 1].x, vertexes[face.v2 - 1].y, vertexes[face.v2 - 1].z)
            glVertex3f(vertexes[face.v3 - 1].x, vertexes[face.v3 - 1].y, vertexes[face.v3 - 1].z)
            glEnd()

    @staticmethod
    def deleteSymbol(file, out_file):
        vertexes = []  # 简化前顶点
        faces = []  # 简化前面片
        with open(file, mode='r', encoding='utf-8') as f:
            with open("noSymbolModel/" + out_file, mode='w', encoding='utf-8') as newFile:
                line = f.readline()
                while line:
                    values = line.split()
                    if values == [] or values[0] == "#" or values[0] == "vt":
                        line = f.readline()
                        continue
                    if values[0] == "v":  # 读取顶点
                        vertex = Vertex(values[1], values[2], values[3])
                        vertex.x = float(values[1])
                        vertex.y = float(values[2])
                        vertex.z = float(values[3])
                        vertexes.append(vertex)
                    elif values[0] == "f":  # 读取面片
                        face = Face(values[1].split('/')[0], values[2].split('/')[0], values[3].split('/')[0])
                        face.v1 = int(values[1].split('/')[0])
                        face.v2 = int(values[2].split('/')[0])
                        face.v3 = int(values[3].split('/')[0])
                        faces.append(face)
                    line = f.readline()
                for v in vertexes:
                    newFile.write("v " + str(v.x) + " " + str(v.y) + " " + str(v.z) + "\n")
                for face in faces:
                    newFile.write("f " + str(face.v1) + " " + str(face.v2) + " " + str(face.v3) + "\n")

    @staticmethod
    def vertex_clustering_simplify(file, out_file):

        start = time.clock()
        vertexes = []  # 简化前顶点
        faces = []  # 简化前面片
        with open(file, mode='r', encoding='utf-8') as f:
            with open("newModel/" + out_file, mode='w', encoding='utf-8') as newFile:
                line = f.readline()
                while line:
                    values = line.split()
                    if values == [] or values[0] == "#" or values[0] == "vt":
                        line = f.readline()
                        continue
                    if values[0] == "v":  # 读取顶点
                        vertex = Vertex(values[1], values[2], values[3])
                        vertex.x = float(values[1])
                        vertex.y = float(values[2])
                        vertex.z = float(values[3])
                        vertexes.append(vertex)
                    elif values[0] == "f":  # 读取面片
                        face = Face(values[1].split('/')[0], values[2].split('/')[0], values[3].split('/')[0])
                        face.v1 = int(values[1].split('/')[0])
                        face.v2 = int(values[2].split('/')[0])
                        face.v3 = int(values[3].split('/')[0])
                        faces.append(face)
                    line = f.readline()
                new_faces = []  # 新面片
                new_vertex_dict = {}  # 记录顶点的哈希值对应的点的序号
                old_to_new_dict = {}  # 旧顶点对应的新顶点
                new_face_set = set()  # 记录面片字符串的集合
                count = 0
                for vertex in vertexes:
                    count = count + 1
                    x = int(math.floor(vertex.x)/3)

                    y = int(math.floor(vertex.y)/3)

                    z = int(math.floor(vertex.z)/3)

                    new_vertex_hash = str(x * 137) + str(y * 149) + str(z * 163)  # 计算hash
                    if new_vertex_dict.get(new_vertex_hash, -1) == -1:  # 不存在则加入
                        new_vertex_dict[new_vertex_hash] = len(new_vertex_dict) + 1
                        old_to_new_dict[count] = len(new_vertex_dict)
                        newFile.write("v " + str(vertex.x) + " " + str(vertex.y) + " " + str(vertex.z) + "\n")
                    else:
                        old_to_new_dict[count] = new_vertex_dict[new_vertex_hash]
                print(str(len(old_to_new_dict)) + " " + str(len(new_vertex_dict)))
                for face in faces:
                    new_face = Face(old_to_new_dict[face.v1], old_to_new_dict[face.v2],
                                    old_to_new_dict[face.v3])
                    new_face.v1 = old_to_new_dict[face.v1]
                    new_face.v2 = old_to_new_dict[face.v2]
                    new_face.v3 = old_to_new_dict[face.v3]
                    new_face_str = str(new_face.v1) + str(new_face.v2) + str(new_face.v3)
                    if new_face_str not in new_face_set:  # 不存在相同的面则加入
                        new_face_set.add(new_face_str)
                        #if new_face.v1 != new_face.v2 and new_face.v1 != new_face.v3 and new_face.v2 != new_face.v3:  # 三点都在不同块内再加入新的面中
                        if not new_face.v1 == new_face.v2 == new_face.v3:  # 如果三个点都在同一立方体内，则简化。即使有两个点在同一方块内也不简化
                            new_faces.append(new_face)
                            newFile.write(
                                "f " + str(new_face.v1) + " " + str(new_face.v2) + " " + str(new_face.v3) + "\n")
                end = time.clock()
                print('运行时间 : %s 秒' % (end - start))
                print(str(len(faces)) + " " + str(len(new_faces)) + " " + str(
                    float(len(faces) - len(new_faces)) / float(len(faces))))

    @staticmethod
    def search_data(data_values, coordinates, skip, data_type):
        for d in data_values:
            if d == skip:
                continue
            if data_type == 'float':
                coordinates.append(float(d))
            elif data_type == 'int':
                coordinates.append(int(d) - 1)

    @staticmethod  # sorted vertex buffer for use with glDrawArrays function
    def create_sorted_vertex_buffer(indices_data, vertices, textures, normals):
        for i, ind in enumerate(indices_data):
            if i % 3 == 0:  # sort the vertex coordinates
                start = ind * 3
                end = start + 3
                ObjLoader.buffer.extend(vertices[start:end])
            elif i % 3 == 1:  # sort the texture coordinates
                start = ind * 2
                end = start + 2
                ObjLoader.buffer.extend(textures[start:end])
            elif i % 3 == 2:  # sort the normal vectors
                start = ind * 3
                end = start + 3
                ObjLoader.buffer.extend(normals[start:end])

    @staticmethod  # TODO unsorted vertex buffer for use with glDrawElements function
    def create_unsorted_vertex_buffer(indices_data, vertices, textures, normals):
        num_verts = len(vertices) // 3

        for i1 in range(num_verts):
            start = i1 * 3
            end = start + 3
            ObjLoader.buffer.extend(vertices[start:end])

            for i2, data in enumerate(indices_data):
                if i2 % 3 == 0 and data == i1:
                    start = indices_data[i2 + 1] * 2
                    end = start + 2
                    ObjLoader.buffer.extend(textures[start:end])

                    start = indices_data[i2 + 2] * 3
                    end = start + 3
                    ObjLoader.buffer.extend(normals[start:end])

                    break

    @staticmethod
    def show_buffer_data(buffer):
        """arr=[]
        arr.append(1)
        arr.append(2)
        arr.append(3)
        arr.append(4)
        arr.append(5)
        arr.append(6)

        print(arr[1:5])"""

        print(len(buffer))
        '''for i in range(len(buffer) // 8):
            start = i * 8
            end = start + 8
            print(buffer[start:end])'''

    @staticmethod
    def load_model(file, sorted=True):
        vert_coords = []  # will contain all the vertex coordinates
        tex_coords = []  # will contain all the texture coordinates
        norm_coords = []  # will contain all the vertex normals

        all_indices = []  # will contain all the vertex, texture and normal indices
        indices = []  # will contain the indices for indexed drawing

        with open(file, 'r') as f:
            line = f.readline()
            while line:
                if str.find(line, '\\') != -1:
                    line = line + f.readline()
                    line = str.replace(line, '\\', ' ')
                values = line.split()
                if values == []:
                    line = f.readline()
                    continue
                if values[0] == 'v':
                    ObjLoader.search_data(values, vert_coords, 'v', 'float')
                elif values[0] == 'vt':
                    ObjLoader.search_data(values, tex_coords, 'vt', 'float')
                elif values[0] == 'vn':
                    ObjLoader.search_data(values, norm_coords, 'vn', 'float')
                elif values[0] == 'f':
                    for value in values[1:]:
                        val = value.split('/')
                        ObjLoader.search_data(val, all_indices, 'f', 'int')
                        indices.append(int(val[0]) - 1)

                line = f.readline()
        all_indices = all_indices - np.min(all_indices)
        if sorted:
            # use with glDrawArrays
            ObjLoader.create_sorted_vertex_buffer(all_indices, vert_coords, tex_coords, norm_coords)
        else:
            # use with glDrawElements
            ObjLoader.create_unsorted_vertex_buffer(all_indices, vert_coords, tex_coords, norm_coords)

        # ObjLoader.show_buffer_data(ObjLoader.buffer)

        buffer = ObjLoader.buffer.copy()  # create a local copy of the buffer list, otherwise it will overwrite the static field buffer
        ObjLoader.buffer = []  # after copy, make sure to set it back to an empty list

        return np.array(indices - np.min(indices), dtype='uint32'), np.array(buffer, dtype='float32')
