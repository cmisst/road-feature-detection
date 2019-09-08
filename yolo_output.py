def analyze_yolo(lines):
    assert(len(lines)>=2)
    assert(lines.pop(-1)=='Enter Image Path: ')
    lines.pop(0)
    for i in range(len(lines)):
        lines[i] = lines[i].split(sep=':', maxsplit=1)[0]
    d = dict.fromkeys(lines)
    if 'traffic light' in d:
        return 1
    elif 'stop sign' in d:
        return 2
    else:
        return 0

if __name__ == "__main__":
    for i in range(4):
        f = open('tmpout'+str(i)+'.log', 'r')
        print(analyze_yolo(f.readlines()))
        f.close()