import clr
import sys
import time
import numpy as np

clr.AddReference('ttInterface')

import TimeTag
import pyexp.TimeTagUtility.NumpyNetTypeConverter as nnConv

ttInterface = None
ttResolution = 0.0
ttReader = None
totalRunTime = 0.0
                          
def TimeTagInitialize():
    global ttResolution, ttInterface, ttReader

    if ttInterface == None:
        ttInterface = TimeTag.TTInterface()
    if(not ttInterface.IsOpen()):
        ttInterface.Open(1)
    
    if(ttInterface.IsOpen()):
        ttReader = ttInterface.GetReader()
        ttResolution = ttInterface.GetResolution()
        #For UQDevice Timetager, reset the delay set inside
        #the hardware according to resolution, to get no delay coData
        if(ttResolution<1e-10):
            SetDelay([0]*8)
        else:
            SetDelay([0]*16)
    else:
        raise ValueError('Device is not open')

def SetDelay(delay):
    for i,d in enumerate(delay):
        ttInterface.SetDelay(i+1,d)

def Start():
    if ttReader != None:
        ttReader.StartTimetags()
        return True
    return False

def Stop():
    if ttReader != None:
        ttReader.StopTimetags()

def ResetTagReader():
    Stop()
    Start()

def Close():
    global totalRunTime
    if ttInterface != None:
        ttInterface.Close()
    totalRunTime = 0.0

def GetTimeTag(refreshTime):
    global totalRunTime
    countList = []
    t = 0
    chArray = np.empty(0,np.dtype('uint8'))
    tagArray = np.empty(0,np.dtype('int64'))

    tic = time.clock()
    while(t <= refreshTime and time.clock()-tic <= refreshTime):
        count,channel,tag = ttReader.ReadTags([], [])
        if(count>0):
            chArray = np.concatenate((chArray, nnConv.asNumpyArray(channel,count)))
            tagArray = np.concatenate((tagArray, nnConv.asNumpyArray(tag,count)))
            countList.append(count)
            t = (tagArray[-1]-tagArray[0])*ttResolution

    if(chArray.shape[0] == 0):
        t = refreshTime
    else:
        # reset timetag reading to 0 when [TotalTags > 2 million] and [last tag > refreshTime]
        if(tagArray[-1]*ttResolution > refreshTime and ttReader.TotalTags > 2e6):
            ResetTagReader()

    totalRunTime += t
    return chArray, tagArray, t


def GetTimeTagOneReadout():
    global totalRunTime

    t = 0
    chArray = np.empty(0,np.dtype('uint8'))
    tagArray = np.empty(0,np.dtype('int64'))

    tic = time.clock()
    time.sleep(0.05)
    count,channel,tag = ttReader.ReadTags([], [])
    if(count>0):
        chArray = nnConv.asNumpyArray(channel,count)
        tagArray = nnConv.asNumpyArray(tag,count)
        t = (tagArray[-1]-tagArray[0])*ttResolution
    else:
        t = time.clock()-tic

    ## reset timetag reading to 0 when [TotalTags > 2 million]
    #if(ttReader.TotalTags > 2e6):
    #    ResetTagReader()

    totalRunTime += t
    return chArray, tagArray, t


def GetDeviceErrorText():
    text = ttInterface.GetErrorText(ttInterface.ReadErrorFlags())
    return text