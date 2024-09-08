# Source: https://stackoverflow.com/questions/2846653/how-to-use-threading-in-python

import threading
import queue
from time import sleep
import numpy as np

class ProcessingThread(threading.Thread):

    def __init__(self,qIn):
        super(ProcessingThread, self).__init__()
        self.qIn=qIn
        self.stoprequest = threading.Event()
        self.procCount = 0

    def run(self):

        assert isinstance(self.qIn, queue.Queue)

        while not self.stoprequest.isSet():
            try:
                cData = self.qIn.get(True, 1)
                self.procCount += 1
            except queue.Empty:
                sleep(0.0001)
                continue


    def join(self, timeout=None):
        self.stoprequest.set()
        super(ProcessingThread, self).join(timeout)




class AcquisitionThread(threading.Thread):

    def __init__(self,qOut):
        super(AcquisitionThread, self).__init__()
        self.qOut=qOut
        self.stoprequest = threading.Event()
        self.acCount = 0

    def run(self):

        assert isinstance(self.qOut, queue.Queue)

        while not self.stoprequest.isSet():
            data = np.random.rand(2048, 500) # size of BScan
            self.qOut.put(data)
            self.acCount += 1
            sleep(0.001)


    def join(self, timeout=None):
        self.stoprequest.set()
        super(AcquisitionThread, self).join(timeout)



ipc = queue.Queue()

procTh = ProcessingThread(ipc)
procTh.start()

acqTh = AcquisitionThread(ipc)
acqTh.start()

secs = 10
sleep(secs)


acqTh.join()
procTh.join()

print( "BScans per second: %.1f / %.1f" % (acqTh.acCount/secs, procTh.procCount/secs))
print( "MB per second: %.1f" % (procTh.procCount/secs * 2048 * 500 * 4 / 1000000) )




