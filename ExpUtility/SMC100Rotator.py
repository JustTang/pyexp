import sys
sys.path.append(r'C:\Program Files (x86)\Newport\MotionControl\SMC100\Bin')

import time

import clr
clr.AddReference("Newport.SMC100.CommandInterface")
import CommandInterfaceSMC100 


instrumentKey = 'COM7'
SMC = None # create a device ref object

def OpenSMC100():
    global SMC
    SMC = CommandInterfaceSMC100.SMC100()
    result = SMC.OpenInstrument(instrumentKey)
    if(not result): return True
    return False



def CloseSMC100():
    if(SMC != None):
        result = SMC.CloseInstrument()
        if(not result): return True
        return False



def Home(address):
    try:
        if(Is_NOT_REFERENCED(address)):
            result, errString = SMC.OR(address,'')
        else:
            raise ValueError("Home: Cannot Home, Check Stage State!")
    except ValueError as e:
        print(e.args[0])
        raise 



def GetAddress(address):
    cAddress,errString = SMC.SA_Get(address,2,'')
    return cAddress



def Is_NOT_REFERENCED(address):
    NOT_REFERENCED_Code = ["0A","0B","0C","0D","0E","0F","10","11"]
    # Get controller state
    result, errorCode, controllerState, errString = SMC.TS(address,'','','') 
    if result == 0:
        for code in NOT_REFERENCED_Code:
            if(code == controllerState):
                return True

        return False
    else:
        raise ValueError("Is_NOT_REFERENCED: %s"%errString)
    


def IsReady (address):
    # Get controller state
    result, errorCode, controllerState, errString = SMC.TS(address,'','','') 
    if result == 0:
        if(controllerState == "32" or
            controllerState == "33" or
            controllerState == "34" or
            controllerState == "35"):
            return True

        return False

    return False
        



def GetCurrentPosition(address):	
    result, position, errString = SMC.TP(address, 0.0, '') 
    return result, position, errString



def AbsoluteMove(address, targetPosition):
	# Execute an absolute motion	
    result, errStringMove = SMC.PA_Set(address, targetPosition, '')
    return result, errStringMove




def MultiAbsoluteMove(addressList, targetPositionList):
    try:
        for i,address in enumerate(addressList):
            result, errStringMove = AbsoluteMove(address, targetPositionList[i])
            if(result != 0):
                raise ValueError(
                    "MultiAbsoluteMove at address[%d] Error: %s"%(address,errStringMove))
    except ValueError as e:
        print(e.args[0])
        raise




def MultiWaitEndOfMotion(addressList):
    if(len(addressList) == 0):
        return
    try:
        isRaiseError = False
        errorAddress = 0
        errStr = ''
        ControllerState = "28"
        readyAddress = []
        while(ControllerState == "28"):
            ControllerState = "33"
            for address in addressList:
                #pass check the ready controller
                if not(address in readyAddress):
                    result, errorCode, conState, errString = SMC.TS(address,'','','')
                    if(conState == "33"):
                        readyAddress.append(address)
                    time.sleep(0.02)
                    if(conState == "28"):
                        ControllerState = "28"
                    if(result != 0):
                        isRaiseError = True
                        errorAddress = address
                        errStr = errString
                        #Controller state readout error caused by 
                        #data transmission loss when using RS232 to USB port adapter 
                        if(conState == ''):
                            ControllerState = "28"
            time.sleep(0.1)

        if(isRaiseError):
            raise ValueError(
                "MultiWaitEndOfMotion at address[%d] Error: %s"%(errorAddress,errStr))                

    except ValueError as e:
        print(e.args[0])
        raise



def MultiAbsoluteMoveAndWait(addressList, targetPositionList):
    MultiAbsoluteMove(addressList, targetPositionList)
    MultiWaitEndOfMotion(addressList)



def MultiGetCurrentPosition(addressList):
    positionList = []
    for address in addressList:
        result,position,errString = GetCurrentPosition(address)
        if(result!=0):
            print(errString)
            position = 0.0
        positionList.append(position)

    return positionList





