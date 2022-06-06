__docformat__ = "google"
# build doc : pdoc -o D:\Dropbox\Run4_transitoire\Documentation D:\Dropbox\Run4_transitoire\Functions
# add project folder path to Anaconda\Lib\site-packages in python38.pth 

"""
    This module holds the code to prepare data and interact with Feflow through the official ifm38 library
"""


import gc
import time
from os import chdir, listdir, remove
import os.path
from os.path import isfile, join

import pandas as pd
import numpy as np

# Path to the FeFlow binaries has to bet set before importing ifm38
#chdir(r'C:\\Program Files\\DHI\\2021\\FEFLOW 7.4\\bin64')

#   To import ifm38 from anaconda prompt, it's needed to add in the folder
#   Anaconda\Lib\site-packages a file named python38.pth with the direct path
#   to feflow, in my case : C:\\Program Files\\DHI\\2021\\FEFLOW 7.4\\bin64
#import ifm38 as ifm


""" ------------------------- Defining Feflow Callbacks Function to act on simulator ------------------------- """


def preEnterSimulator(doc):

    """
    Assigning parameters to Feflow before launching any simulation. Parameters are passed
    by global variables has defined in Feflow documentation. In this code, we send permKx,
    permKz and recharge issued from geostatistical simulations. 

    Args:
        doc (PyFeflowDoc): Feflow Document imported with ifm.loadDocument(fem_file)

    Raises:
        ValueError: Wrong number of elements in _PERM_KX
        ValueError: Wrong number of elements in _PERM_KY
        ValueError: Wrong number of elements in _PERM_KZ
        ValueError: Wrong number of recharge in _RECHARGE
        ValueError: Wrong number of nodes while importing steady heads
    """
    
    if (_ELEMENT_NUMBER!=len(_PERM_KX)):
        raise ValueError("Wrong number of elements in permKx")

    # To modify when switching to full anisotropic k fields
    if (_ELEMENT_NUMBER!=len(_PERM_KX)):
        raise ValueError("Wrong number of elements in permKy")

    if (_ELEMENT_NUMBER!=len(_PERM_KZ)):
        raise ValueError("Wrong number of elements in permKz")

    #if (len(doc.getParamValues(ifm.Enum.P_IOFLOW))!=len(_RECHARGE)):
    #    raise ValueError("Wrong number of recharge zones")

    # In this case, Kx = Ky, _PERM_KX is assigned to P_CONDX and P_CONDY

    doc.setParamValues(ifm.Enum.P_CONDX,_PERM_KX,0)
    doc.setParamValues(ifm.Enum.P_CONDY,_PERM_KX,0)
    doc.setParamValues(ifm.Enum.P_CONDZ,_PERM_KZ,0)
    
    #doc.setParamValues(ifm.Enum.P_IOFLOW,_RECHARGE,0)
    doc.setParamValues(ifm.Enum.P_COMP,_SS,0)

    # In transient case, we want to set hydraulic head simulated in steady state simulation
    # as initial hydraulic heads to speed up simulation time.

    if _TRANSIENT == True :
        
        #steady_heads = np.load(_PATH_SAVE+"/temp/steady_heads_"+str(_MODEL_NUMBER)+".npy")
        steady_heads = _TEMP_HEAD 
        
        if (doc.getNumberOfNodes()!=len(steady_heads)):
            raise ValueError("Wrong number of nodes while importing steady heads")


        #doc.setParamValues(ifm.Enum.P_HEAD,steady_heads.tolist(),0) 
        doc.setParamValues(ifm.Enum.P_HEAD,steady_heads,0) 

        del steady_heads
        

def postTimeStep(doc):

    """
    Assigning parameters to Feflow before launching any simulation. Parameters are passed
    by global variables has defined in Feflow documentation. In this code, we send permKx,
    permKz and recharge issued from geostatistical simulations. 

    Args:
        doc (PyFeflowDoc): Feflow Document imported with ifm.loadDocument(fem_file)
    """

    # In steady state we stored the hydraulic heads simulated at each nodes to use them
    # as initial hydraulic head in transient simulation
    if _STEADY == True :
        global _TEMP_HEAD
        steady_heads = doc.getParamValues(ifm.Enum.P_HEAD)

        #np.save(_PATH_SAVE+"/temp/steady_heads_"+str(_MODEL_NUMBER),steady_heads)
        _TEMP_HEAD = steady_heads

        array_sum = np.sum(steady_heads)
        array_has_nan = np.isnan(array_sum)

        if np.max(steady_heads) > 100 or np.min(steady_heads) < 1 or array_has_nan :
            global _STEADY_CONVERGENCE
            _STEADY_CONVERGENCE = False
    

        del steady_heads


        global _STEADY_HEAD
        temp_heads = np.zeros(len(_WELL))

        for well in range(len(_WELL)) :
            temp_heads[well] = doc.getResultsFlowHeadValueAtXYZ(_WELL[well][0], _WELL[well][1], _WELL[well][2])

        _STEADY_HEAD.append(temp_heads)



    else :

        # well + 1 to store an additionnal columns with time of simulation
        temp_heads = np.zeros(len(_WELL)+1)


        _time = doc.getAbsoluteSimulationTime()

        print("Time step :"+str(np.round(_time,1))+"d, Total ellapsed time : "+str(int(time.time()-_TIME))+"s")

        global _HEAD
        for well in range(len(_WELL)) :
            temp_heads[0] = _time
            temp_heads[well+1] = doc.getResultsFlowHeadValueAtXYZ(_WELL[well][0], _WELL[well][1], _WELL[well][2])
        _HEAD.append(temp_heads)

        
        # Put a treshold to delete non converging simulations exceeding a given time (mean converging time on succesfull simulation + 200 sec buffer) 
        if (_time < 3 and (time.time()-_TIME)> 550) or (_time < 6 and (time.time()-_TIME)> 800) or (_time < 11 and (time.time()-_TIME)> 1000) or (_time < 20 and (time.time()-_TIME)> 1200) or (time.time()-_TIME> 1600) :

            doc.stopSimulator()
            doc.closeDocument()
            global _TRANSIENT_CONVERGENCE
            _TRANSIENT_CONVERGENCE = False
        

def check_launch_feflow(path_data: str,
                    fem_steady_model: str,
                    fem_transient_model: str,
                    assim_number: int,
                    model_number: int) -> None:

    """    
    To avoid redoing lengthy simulation on already done one, we check the list of simulation to do
    versus the one already done. We also checked the existence of folders.

    Args:
        path_data (str): Abolute path to the Data folder contained in the project
        fem_steady_model (str): Abolute path to the Feflow steady state model
        fem_transient_model (str): Abolute path to the Feflow transient state model
        assim_number (int): Actual assimilation step in the multiple data assimilation process
        model_number (int): Number of the model to be run in the actual simulation
        
    Raises:
        ValueError: Individual_Update folder missing in Data/Assim_X/ folder
        ValueError: Head folder missing in Data/Assim_X/ folder
    """

    str_model_number = str(model_number)
    str_assim_number = str(assim_number)

    temp_filename = "assim_"+str_assim_number+"_model_"+str_model_number+".npy"

    path = path_data + "Assim_"+str_assim_number+"/Individual_Update/"
    path_already_done = path_data + "Assim_"+str_assim_number+"/Head/"

    if not os.path.isdir(path):
        raise ValueError("Individual_Update folder missing in Assim_"+str_assim_number+"/Individual_Update/")

    if not os.path.isdir(path_already_done):
        raise ValueError("Head folder missing in Assim_"+str_assim_number+"/Head/")


    files = [f for f in listdir(path) if isfile(join(path, f))]
    files_already_done = [f for f in listdir(path_already_done) if isfile(join(path_already_done, f))]

    if "head_"+str(model_number)+".npy" in files_already_done :
        print("Loading succesfull, permeability field "+str_model_number+" already run")

    else :

        temp_filename = "assim_"+str_assim_number+"_model_"+str_model_number+".npy"

        if temp_filename in files :

            print("Loading succesfull, permeability field "+str_model_number+" being run")

            launch_feflow(path_data,
                                fem_steady_model,
                                fem_transient_model,
                                assim_number,
                                model_number)

        else :
            print("/!\ /!\ Loading unsuccesfull, permeability field assim_"+str_assim_number+"_model_"+\
                str_model_number+".npy is missing.\n Could be normal in case \
            of dropped non converging model /!\ /!\\")



def launch_feflow(path_data: str,
                    fem_steady_model: str,
                    fem_transient_model: str,
                    assim_number: int, 
                    model: int) -> None:
    """
    Functions to assign parameters to Feflow and run the simulation directly from Python.
    A lot of internal global variable are defined in accordance with Feflow documentation.
    They follow a naming scheme to minimize risk of interference with other python file :
    _NAME_OF_INTERNAL_GLOBAL.

    First step is to assign parameters to the steady state Feflow model and running it. Hydraulic 
    heads results are then stored as initial condition from the transient model run afterward. 

    Args:
        path_data (str): Abolute path to the Data folder contained in the project
        fem_steady_model (str): Abolute path to the Feflow steady state model
        fem_transient_model (str): Abolute path to the Feflow transient state model
        assim_number (int): Actual assimilation step in the multiple data assimilation process
        model_number (int): Number of the model to be run in the actual simulation

    Raises:
        ValueError: Mising steady state Feflow model, check the name or file
        ValueError: Mising transient state Feflow model, check the name or file
    """

    
    if not os.path.isfile(fem_steady_model):
        raise ValueError("Mising steady state Feflow model, check the name or file")

    if not os.path.isfile(fem_transient_model):
        raise ValueError("Mising transient state Feflow model, check the name or file")

    

    global _ELEMENT_NUMBER
    global _K_ZONE
    global _RECHARGE_ZONE
    global _UPDATED_ELEMENT
    global _PERM_KX
    global _PERM_KZ
    global _RECHARGE
    global _SS
    global _PATH
    global _PATH_SAVE
    global _FEM_FILE
    global _STEADY
    global _TRANSIENT
    global _HEAD
    global _STEADY_HEAD
    global _WELL
    global _TIME
    global _MODEL_NUMBER
    global _STEADY_CONVERGENCE
    global _TRANSIENT_CONVERGENCE
    global _TEMP_HEAD

    _MODEL_NUMBER = model

    _TIME = time.time()

    # Loading all relevant files, Feflow model, updated parameters, and the kZone and recharge zone
    # linking the global parameters to the element number in the Feflow model.
    _FEM_FILE = fem_steady_model
    doc = ifm.loadDocument(_FEM_FILE)

    _ELEMENT_NUMBER = doc.getNumberOfElements()
    

    _UPDATED_ELEMENT = np.load(path_data+"/FEFLOW_Model/updated_element.npy")
    
    # Elements number count starts at 1 in feflow, check the consitency with input updated_element.npy indexing
    _UPDATED_ELEMENT = (_UPDATED_ELEMENT - 1).astype(int)

    _PATH = path_data+"Assim_"+str(assim_number)+"/Individual_Update/"
    _PATH_SAVE = path_data+"Assim_"+str(assim_number)+"/"

    # _HEAD is fullfilled in the postTimeStep() function called by Feflow
    _HEAD = []
    _STEADY_HEAD = []  
    _TEMP_HEAD = []

    # Coordinates of all observations point used in the model for assimilation 
    # and calibration purpose. They are monitored at each time of simulation.
    _WELL = pd.read_csv(path_data+"/FEFLOW_Model/well_location.inp")

    _WELL = np.vstack((_WELL.X.values-doc.getOriginX(),
                       _WELL.Y.values-doc.getOriginY(),
                       _WELL.Z.values-doc.getOriginZ())).T

    # updated_parameter are coming from the assimilation process, in this model, all data except 
    # the last 47 are local element, with a value for each cell in the model. They are called 
    # local_updated_parameter for this reason. 
    # The last 47 parameters assimilated are global parameters, influencing several cells in 
    # Feflow model through kZones. They are treated separately to bring back kZones to elements.
    # On the 45 parameters, the first 15 are related to kx, the next 15 to kr (ratio of kz to kx),
    # the last 15 are saturation coefficient in different surface zones.


    elementsCoordinates = pd.read_pickle(path_data+"/FEFLOW_Model/feflow_element")
    
    _K_ZONE = elementsCoordinates.K_zone.values

    parameter_name = np.unique(elementsCoordinates.K_zone.values)


    updated_parameter = np.load(path_data+"Assim_"+str(assim_number)+"/Individual_Update/assim_"+\
            str(assim_number)+"_model_"+str(model)+".npy")

    global_parameters = len(updated_parameter) - _ELEMENT_NUMBER 
    
    local_updated_parameter = updated_parameter[:-global_parameters]
    global_updated_parameters = updated_parameter[-global_parameters:]

    kx = np.array(doc.getParamValues(ifm.Enum.P_CONDX))

    """
    We will let the kx value set outside of the local update

    # First we applied permeability coming from global variable, by applying global 
    # value to all elements, local elements will replace these global values where
    # they are defined.
    i = 0
    for name in parameter_name :
        # Feflow takes permeabilty as m/d, in case of m/s, multiplying by 3600 * 24.
        kx[np.where(_K_ZONE==name)] = global_updated_parameters[i] *3600*24
        i +=1


    """

    kx[_UPDATED_ELEMENT] = local_updated_parameter[_UPDATED_ELEMENT] 

    ### ky = kx, no need to add another vector in memory

    kz_temp = np.zeros(_ELEMENT_NUMBER)
    kz = np.array(doc.getParamValues(ifm.Enum.P_CONDZ))

    # First we applied permeability coming from global variable, by applying global 
    # value to all elements, local elements will replace these global values where
    # they are defined.
    i = 15
    for name in parameter_name :
        # Same logic, but kz is defined as a ratio of conductivity to kx. 
        kz_temp[np.where(_K_ZONE==name)] = kx[np.where(_K_ZONE==name)] / global_updated_parameters[i]
        i += 1

    kz[_UPDATED_ELEMENT] = kz_temp[_UPDATED_ELEMENT] 

    ss_temp = np.zeros(_ELEMENT_NUMBER)
    ss = np.array(doc.getParamValues(ifm.Enum.P_COMP))
    i = 30
    for name in parameter_name :
        ss_temp[np.where(_K_ZONE==name)] =  global_updated_parameters[i]
        i += 1

    ss[_UPDATED_ELEMENT] = ss_temp[_UPDATED_ELEMENT] 

    # Feflow is unable to deal with direct numpy array, he needs list
    _PERM_KX = kx.tolist()
    _PERM_KZ = kz.tolist()
    _SS = ss.tolist()

    # To reduce memory impact in case of big Feflow model, we free memory as much as possible
    del _K_ZONE
    del _UPDATED_ELEMENT
    del kx
    del kz
    del kz_temp
    del ss
    del ss_temp
    #del kr
    del updated_parameter
    del local_updated_parameter
    del global_updated_parameters
    del parameter_name

    
    # We need to specify in which kind of simulation we are
    _STEADY = True
    _TRANSIENT = False
    _STEADY_CONVERGENCE = True

    doc.startSimulator()
    doc.stopSimulator()

    doc.saveDocument(fem_steady_model[:-4]+"_temp.fem")

    doc.closeDocument()

    print("Steady simulation "+str(model)+" done in "+str(int(time.time()-_TIME))+"s")

    temp_time = _TIME
    # Reset timer for transient simulation
    _TIME = time.time()

    if _STEADY_CONVERGENCE :
        print("Model has converged")
        _FEM_FILE = fem_transient_model

        _STEADY_HEAD = np.array(_STEADY_HEAD)

        # Steady Head used to debug, to check the convergence of the model before next step
        np.save(_PATH_SAVE+"Steady_head/steady_head_"+str(model)+'.npy',_STEADY_HEAD)

        doc = ifm.loadDocument(_FEM_FILE)

        _STEADY = False
        _TRANSIENT = True
        _TRANSIENT_CONVERGENCE = True

        doc.startSimulator()

        if _TRANSIENT_CONVERGENCE :
            doc.stopSimulator()

            doc.closeDocument()

            _HEAD = np.array(_HEAD)

            np.save(_PATH_SAVE+"Head/head_"+str(model)+'.npy',_HEAD)

        else :
            print("Convergence failed with this permeability field")
            print("Deletion of the permeability field")

            os.remove(path_data+"Assim_"+str(assim_number)+"/Individual_Update/assim_"+\
                str(assim_number)+"_model_"+str(model)+".npy")

    else :
        print("Convergence failed with this permeability field")
        print("Deletion of the permeability field")

        os.remove(path_data+"Assim_"+str(assim_number)+"/Individual_Update/assim_"+\
            str(assim_number)+"_model_"+str(model)+".npy")

    del _HEAD
    del _WELL
    del _STEADY
    del _TRANSIENT
    del _ELEMENT_NUMBER
    del _TEMP_HEAD

    # Using a try before deleting these vectors, if Feflow hasn't deallocating them properly
    try:
        del _PERM_KX
    except :
        pass

    try:
        del _PERM_KZ
    except :
        pass

    try:
        del _SS
    except :
        pass
    # try:
    #     del _RECHARGE
    # except :
    #     pass
    

    print("Total simulation "+str(model)+" done in "+str(int((time.time()-temp_time)//60))+" minutes")

    del _TIME

    gc.collect()
