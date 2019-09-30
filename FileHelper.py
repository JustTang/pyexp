import os,sys
import errno
import pickle
import datetime


def mkDirs(name:str, mode = 511):
    if(not os.path.exists(name)):
        try:
            os.makedirs(name,mode)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise



def GetFilePath(folder,fileName):
    '''return: 
            Path: "projectFolder\folder\fileName" '''
    projectFolder = GetProjectFolderPath()
    folder = folder + "\\"
    dataFolder = os.path.join(projectFolder, folder)
    mkDirs(dataFolder)
    return os.path.join(dataFolder, fileName)



def GetProjectFolderPath():
    return sys.path[0]



def GetNowTimeStr():
    '''return:
            datetime.now(): "yymmddHHMMSS" '''
    return datetime.datetime.now().strftime('%y%m%d%H%M%S')



def SaveData2File(data, dataPath):
    try:
        with open(dataPath,'wb') as f:
            pickle.dump(data, f)
        return True
    except Exception as e:
        print(e.args[0])
    return False


def LoadDataFromFile(dataPath):
    try:
        data = None
        with open(dataPath,'rb') as f:
            data=pickle.load(f)
        return data
    except Exception as e:
        print(e.args[0])
    return None

