import subprocess
import os, sys
import time
import pandas as pd



def listdir_abspath(d):
    return [os.path.abspath(os.path.join(d, f)) for f in os.listdir(d)]



class yolo3(object):
    filenames = None
    procs = None
    fw = None
    fr = None
    df = None
    def __init__(self, *args, **kwargs):
        procs = kwargs['procs']
        self.fw = [ open('tmpout'+str(i)+'.log', 'w') for i in range(procs) ]
        self.procs = [ subprocess.Popen(['./darknet', 'detect', 'cfg/yolov3.cfg', 'yolov3.weights'],
            stdin=subprocess.PIPE, stdout=self.fw[i], stderr=self.fw[i], cwd='../yolo-v3/')
            for i in range(procs) ]
        time.sleep(max(5, procs/3*2))  # let darknet load weigths
        self.fr = [ open('tmpout'+str(i)+'.log', 'r+')  for i in range(procs) ]
        [ f.truncate() for f in self.fr ]

        
    def split_task(self):
        self.filenames = sorted(self.filenames)
        self.construct_dataframe()
        self.filenames += self.filenames[-1:] * \
            (len(self.procs) - len(self.filenames) % len(self.procs)) 
        assert(len(self.filenames) % len(self.procs)==0)
        n = len(self.filenames)//len(self.procs)
        self.filenames = [ self.filenames[i:i+n] for i in range(0, len(self.filenames), n) ]
        assert(len(self.filenames)==len(self.procs))


    def run_all(self):
        for i in range(len(self.filenames[0])):
            for p in range(len(self.procs)):
                self.recognize(p, i) # assumed to take long (>20 seconds)
            if(i>0):
                # no results to analyze in first loop
                self.text_analyzer(i-1)
                # logically at the end of the loop
            time.sleep(20) # after 20 seconds, start checking results
            while(min([len(f.readlines()) for f in self.fr])<2):
                [f.seek(0) for f in self.fr]
                time.sleep(1) # check every 1 second for results
        self.text_analyzer(len(self.filenames[0])-1)


    def construct_dataframe(self):
        # TODO: construct dataframe of image uniqnames
        names = [ n.split('/')[-1] for n in self.filenames ]
        pass


    def text_analyzer(self, group):
        # this function needs to run in <1 second
        [f.seek(0) for f in self.fr]
        for p in range(len(self.fr)):
            # TODO: deal with filename to uniqname self.filenames[p][group]
            result = self.analyze_yolo_output(self.fr[p].readlines())
            # TODO: put results in dataframe
            self.fr[p].truncate()
    

    def analyze_yolo_output(self, lines):
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


    def recognize(self, p, n):
        self.procs[p].stdin.write((self.filenames[p][n]+'\n').encode('utf-8'))
        self.procs[p].stdin.flush()



if __name__ == '__main__':
    y = yolo3(procs=4)
    y.filenames = listdir_abspath('../yolo-v3/data/street/')
    y.split_task()
    y.run_all()

    