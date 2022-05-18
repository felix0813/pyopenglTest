model = "./newModel/test1_test.0.01.obj"
newModel = "./newModel/target.obj"
with open(model) as f:
    with open(newModel, mode="w") as target:
        line = f.readline()
        while line:
            values = line.split()
            if values[0] == 'v':
                x = float(values[1]) * 3
                y = 572-float(values[3]) * 3 - 500
                z = 535+535-(498+float(values[2]) * 3)+100
                newValue = "v " + str(x) + " " + str(y) + " " + str(z) + "\n"
                target.write(newValue)
            elif values[0] == 'f':
                target.write(line)
            line = f.readline()
# 37 107 535