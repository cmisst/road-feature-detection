import subprocess
import os, sys
import time



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
        [ f.truncate(0) for f in self.fr ]

        
    def split_task(self):
        names = []
        for n in self.filenames:
            if n.rsplit('.', 1)[1] in ['jpg', 'png']:
                names.append(n)
        self.filenames = sorted(names)
        # construct dictionary of image uniqnames
        self.df = dict.fromkeys([ int(n.split('/')[-1][0:8]) for n in self.filenames ])
        # padding to align
        self.filenames += self.filenames[-1:] * \
            (len(self.procs) - len(self.filenames) % len(self.procs)) 
        assert(len(self.filenames) % len(self.procs)==0)
        n = len(self.filenames)//len(self.procs)
        self.filenames = [ self.filenames[i:i+n] for i in range(0, len(self.filenames), n) ]
        assert(len(self.filenames)==len(self.procs))


    def run_all(self):
        for i in range(len(self.filenames[0])):
            t = time.time()
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
            print('group {} takes time {}'.format(i+1, time.time()-t))
        self.text_analyzer(len(self.filenames[0])-1)


    def text_analyzer(self, group):
        # this function needs to run in <1 second
        [f.seek(0) for f in self.fr]
        for p in range(len(self.fr)):
            uniqname = int(self.filenames[p][group].rsplit('/', 1)[1][0:8])
            assert(uniqname in self.df)
            result = self.analyze_yolo_output(self.fr[p].readlines())
            self.fw[p].truncate(0)
            # put results in dataframe
            if self.df[uniqname] == 1:
                pass
            else:
                self.df[uniqname] = result
        self.write_progress()


    def write_progress(self, all=False, file=None):
        if file is None:
            file = 'progress.log'
        p = open(file, 'w')
        for k, v in self.df.items():
            if v is not None:
                p.writelines(str(k) + ',' + str(v) + '\n')
            elif all:
                p.writelines(str(k) + ', \n')
        p.close()


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
    y = yolo3(procs=32)
    y.filenames = listdir_abspath('../yolo-v3/data/street/')
    y.split_task()
    y.run_all()
