if __name__ == "__main__":
    for i in range(0, 30):
        with open('model/' + str(i) + '.obj', 'r') as f:
            with open('model/' + str(i + 30) + '.obj', 'w') as newF:
                newF.write("o object_" + str(i + 31) + "\n")
                line = f.readline()
                while line:
                    if str.find(line, '\\') != -1:
                        line = line + f.readline()
                        line = str.replace(line, '\\', ' ')
                    values = line.split()
                    if not values:
                        line = f.readline()
                        continue
                    if values[0] == 'v':
                        newF.write('v ')
                        x = 1
                        for tmp in values:
                            if tmp != 'v':
                                if x == 1:
                                    tmp = float(tmp) + 500.0
                                newF.write(str(tmp) + " ")
                                x = x + 1
                    elif values[0] == 'vt':

                        for tmp in values:
                            newF.write(str(tmp) + " ")
                    elif values[0] == 'vn':

                        for tmp in values:
                            newF.write(str(tmp) + " ")
                    elif values[0] == 'f':

                        for tmp in values:
                            newF.write(str(tmp) + " ")
                    newF.write('\n')
                    line = f.readline()
