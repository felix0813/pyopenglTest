import objLoader

model_list = ["model/人顶罐6.23.obj"]
''', "model/小型人鸟.obj", "model/族长面具1一号.obj", "model/祭司人2号.obj",
              "model/祭司人3号.obj", "model/祭祀人1号.obj", "model/青铜树.obj", "model/面具1号.obj", "model/面具6.23.obj",
              "model/鸡6.23.obj", "model/黄金面具6.23.obj", "model/陶器1.obj"]'''
i = 0
for name in model_list:
    i = i + 1
    print(str(i) + ".")
    objLoader.ObjLoader.vertex_clustering_simplify(name, "test" + str(i)+".obj")
