count = 0
for i in range(0, 30):
    print(i)
    file = "./model/"+str(i) + ".obj"
    with open(file, mode='r', encoding='utf-8') as f:
        line = f.readline()
        while line:
            if 'f' in line:
                count = count + 1
            line = f.readline()
print(count)
