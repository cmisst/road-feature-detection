import subprocess
import os, sys
import time
from select import select



def listdir_abspath(d):
    return [os.path.abspath(os.path.join(d, f)) for f in os.listdir(d)]



class yolo3(object):
    filenames = None
    procs = None
    fw = None
    fr = None
    def __init__(self, *args, **kwargs):
        procs = kwargs['procs']
        self.fw = [ open("tmpout"+str(i), "w") for i in range(procs) ]
        self.procs = [ subprocess.Popen(['./darknet', 'detect', 'cfg/yolov3.cfg', 'yolov3.weights'],
            stdin=subprocess.PIPE, stdout=self.fw[i], stderr=self.fw[i], cwd='../yolo-v3/')
            for i in range(procs) ]
        time.sleep(max(5, procs/3*2))  # let darknet load weigths
        self.fr = [ open("tmpout"+str(i), "r+")  for i in range(procs) ]
        [ f.truncate() for f in self.fr ]

        
    
    def split_task(self):
        self.filenames = sorted(self.filenames)
        if len(self.filenames) % len(self.procs)==0:
            group=len(self.filenames)//len(self.procs)
        else:
            group=len(self.filenames)//len(self.procs)+1
        self.filenames = [ self.filenames[i:i+group] for i in range(0, len(self.filenames), group) ]
        assert(len(self.filenames)==len(self.procs))


    def run_all(self):
        for i in range(len(self.filenames[0])):
            for p in range(len(self.procs)):
                if i >= len(self.filenames[p]):
                    pass
                else:
                    self.recognize(p, i)
            for p in range(len(self.fr)):
                print(self.fr[p].readlines())


    def recognize(self, p, n):
        self.procs[p].stdin.write((self.filenames[p][n]+'\n').encode('utf-8'))
        self.procs[p].stdin.flush()



if __name__ == '__main__':
    y = yolo3(procs=4)
    y.filenames = listdir_abspath('../yolo-v3/data/street/')
    y.split_task()
    y.run_all()

    