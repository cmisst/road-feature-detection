def analyze_yolo(lines):
    print(lines)
    return 0

if __name__ == "__main__":
    for i in range(4):
        f = open('tmpout'+str(i)+'.log', 'r')
        analyze_yolo(f.readlines())
        f.close()