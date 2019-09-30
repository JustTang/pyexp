import numpy as np
import time
from numba import jit
import matplotlib.pyplot as plt

import pyexp.TimeTagUtility.TimeTagHelper as ttH
import pyexp.TimeTagUtility.DataFit as dataFit

class Pauli:
    '''Four 2d Pauli Matrices'''
    I = np.array([[1,0],[0,1]])
    X = np.array([[0,1],[1,0]])
    Y = np.array([[0,-1j],[1j,0]])
    Z = np.array([[1,0],[0,-1]])


def H(A):
    '''Hermitian Conjugation'''
    return np.transpose(A).conj()

def angle2U(angle, delta):
    angleR = angle/180*np.pi
    deltaR = delta/180*np.pi
    U1 = [[np.cos(angleR)**2, np.cos(angleR)*np.sin(angleR)],
          [np.cos(angleR)*np.sin(angleR), np.sin(angleR)**2]]
    U2 = [[np.sin(angleR)**2, -np.cos(angleR)*np.sin(angleR)],
          [-np.cos(angleR)*np.sin(angleR), np.cos(angleR)**2]]
    return np.array(U1) + np.exp(1j*deltaR)*np.array(U2)

def U2QHQ(U, h90 = False):
    '''2D Unitary --> WP rotating angles(operation order: left->right)
    params:
        U: numpy.array
        h90 = True: HWP<90
    return:
        qhq: numpy.array
    '''
    
    phase = np.angle(np.linalg.det(U))/2
    u = np.exp(-1j*phase)*U
    A = np.array([[1-1j, 1+1j],
                  [1+1j, 1-1j]])/2
    u = H(A).dot(u).dot(A)
    a = -2*np.angle(u[0,0])
    b = 2*np.angle(u[1,0])
    alpha = (a+b)/2
    gamma = (a-b)/2
    beta = 2*np.angle( u[0,0]*np.exp(1j*a/2) + 1j*u[1,0]*np.exp(-1j*b/2) )
    qhq = np.array([np.pi/4 - gamma/2, -np.pi/4 + (-beta+alpha-gamma)/4, np.pi/4+alpha/2])
    qhq = (qhq % np.pi)/np.pi*180
    if(h90): qhq[1] = qhq[1]%90
    return qhq


def QHQ2U(qhq):
    '''
        operation order: left->right
    '''
    return angle2U(qhq[2],90).dot(angle2U(qhq[1],180)).dot(angle2U(qhq[0],90))



def ParamSphere2U(x):
    '''3 params np.array([alpha, theta, phi]) in spherical coordinates --> 2d Unitary'''
    U = [[np.cos(x[0]) - 1j*np.sin(x[0])*np.cos(x[1]), -( np.sin(x[1])*np.sin(x[2]) + 1j*np.sin(x[1])*np.cos(x[2]) )*np.sin(x[0])],
         [( np.sin(x[1])*np.sin(x[2]) - 1j*np.sin(x[1])*np.cos(x[2]) )*np.sin(x[0]), np.cos(x[0]) + 1j*np.sin(x[0])*np.cos(x[1])]]
    return np.array(U)



def U2ParamSphere(U):
    '''2d Unitary --> 3 params np.array([alpha, theta, phi]) in spherical coordinates'''
    phase = np.angle(np.trace(U))
    #trace(Ua) is real (Ua dim=2)
    Ua = np.exp(-1j*phase)*U
    #alpha: [0, pi/2]
    alpha = np.arccos(np.abs(np.trace(Ua)/2))
    if(alpha == 0):
        return np.array([0.0, 0.0, 0.0])
    
    tempZ = np.trace(Ua.dot(Pauli.Z)) 
    if(tempZ.imag != 0): tempZ = -tempZ.imag
    tempX = np.trace(Ua.dot(Pauli.X)) 
    if(tempX.imag != 0): tempX = -tempX.imag
    tempY = np.trace(Ua.dot(Pauli.Y)) 
    if(tempY.imag != 0): tempY = -tempY.imag
    #theta:[0, pi)
    theta = np.arccos( tempZ/(2*np.sin(alpha)) )

    if(theta == 0):
        return np.abs(np.array([alpha, 0, 0]))

    cosPhi = tempX/(2*np.sin(alpha)*np.sin(theta))
    sinPhi = tempY/(2*np.sin(alpha)*np.sin(theta))
    #phi:[0, 2*pi)
    phi = np.angle(cosPhi+1j*sinPhi)
    if(phi<0):
        phi = 2*np.pi+phi

    return np.abs(np.array([alpha, theta, phi]))


def r2qh(r):
    l = np.linalg.norm(r)
    blochVec = None
    if(l==0):
        return np.random.rand(1,2)*np.array([180,90])
    else:
        blochVec = r/l

    beta = None
    if(r[2] == 0):
        beta = np.pi/2
    else:
        beta = np.arctan(-r[0]/r[2])
    alpha = np.arcsin(r[0]-r[1]*np.cos(beta))
    if( (np.cos(alpha)-(r[2]-r[1]*np.sin(beta)))**2 > 1e-10 ):
        alpha = np.pi-alpha
    qwp = beta/2
    qwp = qwp.real
    hwp = (alpha+2*beta)/4
    hwp = hwp.real
    qh = -np.array([qwp,hwp])/np.pi*180
    qh = np.mod(qh,[180, 90])
    return qh


def DenMat2BlochVec(denMat):
    r1 = np.trace(Pauli.X.dot(denMat))
    r2 = np.trace(Pauli.Y.dot(denMat))
    r3 = np.trace(Pauli.Z.dot(denMat))
    return np.array([r1.real,r2.real,r3.real])
        

def TSP(initPos, posArray:np.ndarray):
    '''Find a route order based on local minimum distance
    return: sortIndex --> route order'''
    N = len(posArray)
    index = np.arange(0,N)
    sortIndex = np.zeros(N, np.dtype('int32'))

    currentPos = initPos.copy()
    posArrayCopy = posArray.copy()
    for i in range(N):
        dist = np.sum( np.abs(posArrayCopy-np.tile(currentPos,(len(posArrayCopy),1))) , 1)
        indexMin = np.argmin(dist)
        sortIndex[i] = index[indexMin]
        currentPos = posArrayCopy[indexMin]
        posArrayCopy = np.delete(posArrayCopy, indexMin, 0)
        index = np.delete(index, indexMin, 0)

    return sortIndex


def BellMeasureProba(denMat):
    basis = np.array([[0,1,1,0],
                      [1,0,0,1],
                      [1,0,0,-1],
                      [0,1,-1,0]])/np.sqrt(2)
    basis = basis.T
    povm = Kraus2Povm(basis)
    proba = [np.trace(denMat.dot(element)) for element in povm]
    return np.abs(np.array(proba))


def Kraus2Povm(basis):
    '''Convert measurement basis to POVM element matrix 
    basis: column vectors in a 2D np.array
    return: list of POVM element matrix'''
    povm = []
    N = basis.shape[1]
    for i in range(N):
        povm.append(basis[:,i:i+1].dot(H(basis[:,i:i+1])))
    return povm



def GetCoincidenceFixCoCount(coAnalyzer, delay, coPattern, coWindow, 
                           coCountPosList, nTotal, nSingle = None):
    if(not nSingle):
        nSingle = nTotal
    # the sum of coincidence counts in one row
    # of [totalCountData] is [nSingle];
    # the part whose sum of coincidence counts is less than [nSingle]
    # is in [countLast];
    totalCountData = np.empty((0,len(coPattern)),np.dtype('int32'))
    countLast = np.zeros(len(coPattern), np.dtype('int32'))
    totalCoCount = 0
    sampleTime = 0

    if(ttH.Start()):
        tic = time.clock()
        i=0
        while(totalCoCount < nTotal):
            chArray, tagArray, t = ttH.GetTimeTagOneReadout()
            coCount = coAnalyzer.GetCoincidenceFixCoCount(
                        chArray, tagArray, delay, coPattern, coWindow,
                        countLast, coCountPosList, nSingle)
            i+=1
            countLast = coCount[-1]
            if(coCount.shape[0]>1):
                totalCountData = np.vstack((totalCountData,coCount[0:coCount.shape[0]-1]))
            totalCoCount = nSingle*totalCountData.shape[0]
            sampleTime += t 
            # handle no signal event
            if(countLast.sum()==0 and time.clock()-tic>10):
                raise ValueError('No input to CoDevice!')

        ttH.Stop()

        if(totalCoCount>nTotal):
            totalCountData = totalCountData[0:int(nTotal/nSingle)]

    else:#Start Error. Return 0 count
        totalCountData = countLast
    return totalCountData, sampleTime



def GetCoincidenceFixTime(coAnalyzer, delay, coPattern, coWindow, time):
    ttH.Start()
    chArray,tagArray,t = ttH.GetTimeTag(time)
    ttH.Stop()
    countData = coAnalyzer.GetCoincidence(
                    chArray,tagArray,delay,coPattern,coWindow)
    return countData


def TimetagSample(coAnalyzer, time, fileFolder):
    ttH.Start()
    chArray,tagArray,t = ttH.GetTimeTag(time)
    ttH.Stop()
    ttH.Close()
    coAnalyzer.SaveSampleTimetagData(chArray,tagArray,fileFolder)



def DelaySearch(coAnalyzer, coChannel, fileFolder,
               searchRange = [0,100], coWindow = 4):
    '''
    Using the sampled timetag data to search the delay between [coChannel];
    The resolution is about the resolution of the CoDevice.

    Args:
        coAnalyzer {pyexp.TimeTagUtility.CoincidenceAnalyzer}
        coChannel {list[2]} -- the 2 channels to coincidence and 
            delay is added to the later one.
        fileFolder {string} -- the file folder with timetag data, 
            the path starts from the project folder.
        searchRange{list[2]} -- must >=0; unit in ns
        coWindow{int} -- CoDevice resolution unit

    Returns:
        delay{float} -- applied on second channel; unit in ns
        coChannel{list[2]} -- maybe reversed
        chArray, tagArray -- for further reusing
    '''
    chArray,tagArray = coAnalyzer.LoadSampleTimetagData(fileFolder)
    #Coarse search: step=1ns, coWindow=3ns
    delays,counts = coAnalyzer.DelayScan(
            chArray, tagArray, coAnalyzer.ns2tagUnit(3),
            coAnalyzer.ns2tagUnit(searchRange), 
            coAnalyzer.ns2tagUnit(1), coChannel)
    maxIndex = np.argmax(counts)
    minIndex = np.argmin(counts)
    minValue = 1 if(counts[minIndex] == 0) else counts[minIndex]
    #use max/min ratio to check the peak
    if(counts[maxIndex]/minValue > 50):     
        searchRange = [delays[maxIndex]-coAnalyzer.ns2tagUnit(2),
                       delays[maxIndex]+coAnalyzer.ns2tagUnit(2)]
        if(searchRange[0]<0):
            searchRange[0] = [0,coAnalyzer.ns2tagUnit(3)]
    else:   #reverse search
        coChannel = coChannel[::-1]
        delays,counts = coAnalyzer.DelayScan(
        chArray, tagArray, coAnalyzer.ns2tagUnit(3),
        coAnalyzer.ns2tagUnit(searchRange), 
        coAnalyzer.ns2tagUnit(1), coChannel)
        maxIndex = np.argmax(counts)
        minIndex = np.argmin(counts)
        minValue = 1 if(counts[minIndex] == 0) else counts[minIndex]
        if(counts[maxIndex]/minValue > 50):    
            searchRange = [delays[maxIndex]-coAnalyzer.ns2tagUnit(2),
                           delays[maxIndex]+coAnalyzer.ns2tagUnit(2)]
            if(searchRange[0]<0):
                searchRange[0] = [0,coAnalyzer.ns2tagUnit(3)]
        else:
            print("Delay Searching is failed. Recheck input data.")
            return
    #Fine search: step=1unit, coWindow=5unit
    delays,counts = coAnalyzer.DelayScan(
            chArray, tagArray, coWindow, searchRange, 1, coChannel)
    maxIndex = np.argmax(counts)

    return (coAnalyzer.tagUnit2ns(delays[maxIndex]), 
            coChannel, chArray, tagArray)



def DelayScanPlot(coAnalyzer, scanRange, chArray, tagArray, coChannel, 
                  color, coWindow = 4, step = 1):
    '''
    Plot fine delay scan peak and use guassian fit to get 
    a better delay pricision[~3mm@78ps?].
    Need to call [DelaySearch] first to get a proper [scanRange].
    '''
    delays,counts = coAnalyzer.DelayScan(
            chArray, tagArray, coWindow,
            coAnalyzer.ns2tagUnit(scanRange), 
            step, coChannel)
    delays = coAnalyzer.tagUnit2ns(delays)
    plt.scatter(delays, counts, c = color, s = 20)
    maxIndex = np.argmax(counts)
    mu = delays[maxIndex]
    hight = counts[maxIndex]
    i = maxIndex
    while(hight-counts[i] < hight/2):
        i += 1
    sigma = (delays[i] - mu)/(2*np.log(2))
    output = dataFit.GaussianFit([mu, sigma, hight], delays, counts)
    output = output[0]
    xFit = np.arange(mu-5, mu+5, 0.01)
    yFit = [dataFit.GaussianFunc(x,output) for x in xFit]
    plt.plot(xFit,yFit,label=r'$\mu = %.5f$'%output[0], c = color)
    return output[0]



def CoincidenceRuler(coAnalyzer, coChannel, fileFolder1, fileFolder2):
    '''
    Get and show the distance change coming from the delay change.
    Need to sample data first by calling [TimetagSample].
    '''
    delay1,coChannel,chArray1,tagArray1 = DelaySearch(coAnalyzer, 
                                                      coChannel, fileFolder1)
    delay2,_,chArray2,tagArray2 = DelaySearch(coAnalyzer, 
                                              coChannel, fileFolder2)
    fig = plt.figure()
    scanRange1 = [delay1-2, delay1+2] if delay1>=2 else [0,delay1+4]
    scanRange2 = [delay2-2, delay2+2] if delay2>=2 else [0,delay2+4]
    fitDelay1 = DelayScanPlot(coAnalyzer, scanRange1, chArray1, 
                              tagArray1, coChannel, 'k')
    fitDelay2 = DelayScanPlot(coAnalyzer, scanRange2, chArray2, 
                              tagArray2, coChannel, 'r')
    plt.show()
    t = fitDelay2-fitDelay1
    distance = t*0.299792458
    return distance,t

