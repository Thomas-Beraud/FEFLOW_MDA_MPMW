__docformat__ = "google"
# build doc : pdoc -o D:\Dropbox\Run4_transitoire\Documentation D:\Dropbox\Run4_transitoire\Functions
# add project folder path to Anaconda\Lib\site-packages in python38.pth 

"""
    This module handle all the observation in a convenient way for the assimilation process
"""

from time import time
import numpy as np
import matplotlib.pyplot as plt
import re
from os import listdir, remove
from os.path import isfile, join, exists
import shutil
from datetime import datetime
import pandas as pd

class Well():
    """
    Well class holds every data with corresponding time step, coordinates and name
    """
    def __init__(self, name, size, x, y, z):
        """
        A Well is initiated through a name, size and coordinates  
        """

        self.name = name
        """
        name (str): name of the well
        """
        self.size = size
        """
        size (int): number of data points for the well
        """
        self.time = [datetime for _ in range(self.size)]
        """
        time (datetime): Each timestep corresponding to a datapoint 
        """
        self.data = np.zeros(self.size)
        """
        data (float): Each datapoint corresponding to a timestep
        """
        self.x = x
        """
        x (float): x coordinate for the well
        """
        self.y = y
        """
        y (float): y coordinate for the well
        """
        self.z = z
        """
        z (float): z coordinate for the well
        """


    def set_data(self,data,time):
        """
        Function to reassign time and data to a Well object

        Args:
            data (float): Each datapoint corresponding to a timestep
            time (datetime): Each timestep corresponding to a datapoint 
        
        Returns:
            None

        Raises:
            Your value array size is different than the one defining this well. Data array should be the same size as self.size

        """

        if len(data) == self.size :
            self.data = data
        if len(time) == self.size :
            self.time = time
        else :
            print("Your value array size is different than the one defining this well \nYour data array should be of size : "+str(self.size)+" ")



    def set_data_index(self,data,time,index):
        """
        Function to reassign only one data or time index in a Well

        Args:
            data (float): Data at the index position in the array to be replaced
            time (datetime): Time at the index position in the array to be replaced
            index (int): index in both array to replaced
        
        Returns:
            None

        Raises:
            Your index is greater than the array size. Max index : self.size

        """

        if self.size >= index :
            self.data[index] = data
            self.time[index] = time
        else :
            print("Your index is greater than the array size. Max index : "+str(self.size)+" ")


    def show_well_description(self) :
        """
        Print all Well informations (name,size,x,y,z)
        """
        print("Name : "+str(self.name)+"")
        print("Size : "+str(self.size)+"")
        print("X : "+str(self.x)+"")
        print("Y : "+str(self.y)+"")
        print("Z : "+str(self.z)+"")

    def show_well_data(self) :
        """
        Print all Well data (time,data)
        """
        print(self.time,self.data)

    def get_well_description(self) :
        """
        Return all Well informations (name,size,x,y,z)

        Args:
            None
        
        Returns:
            name, size, x, y, z
        """

        return(self.name,self.size,self.x,self.y,self.z)


    def get_well_data(self) :
        """
        Return all Well datas (time, data)

        Args:
            None
        
        Returns:
            time, data
        """

        return(self.time,self.data)

    def plot(self,x_size=8,y_size=4):
        """
        Display the datas

        Args:
            x_size (float=8): x dimension of the plot
            y_size (float=4): y dimension of the plot

        Returns:
            None: Display the plot
        """

        plt.figure(figsize=(x_size,y_size))

        plt.plot(self.time,self.data, 'r', label='Measured',linewidth=3)

        plt.xlabel('Time',fontsize=16)
        plt.ylabel('Measure',fontsize=16)

        plt.legend(fontsize=16)

        plt.title(self.name+" \n x:"+str(self.x)+" y:"+str(self.y)+" z:"+str(self.z),fontsize=20)

        plt.show()



class Measure():
    """
    Measure class holds every well measured on the field
    """


    def __init__(self, path):
        """path (str): path to the observation file produced by load_results()"""
        f = open(path,'r')

        self.nbr_obs_points = int(f.readline().partition(';')[0])
        """nbr_obs_points (int): Number of observations points in the model, corresponding to the amount of measures done on the field"""

        self.wells = [Well for i in range(self.nbr_obs_points)]
        """wells (Well): List of Well for each observation point"""

        for i in range(self.nbr_obs_points):
            name = f.readline().partition(';')[0]
            size = int(f.readline().partition(';')[0])
            x = float(f.readline().partition(';')[0])
            y = float(f.readline().partition(';')[0])
            z = float(f.readline().partition(';')[0])

            self.wells[i] = Well(name, size, x, y, z)

            temp_data = np.zeros(size)
            temp_time = [datetime for _ in range(size)]

            for j in range(size):
                temp = (f.readline()).split(',')
                temp_data[j] = float(temp[1])
                temp_time[j] = datetime.strptime(temp[0],"%m/%d/%Y %H:%M:%S")

            self.wells[i].set_data(temp_data,temp_time)

        f.close()


class Observation():
    """
    Observation class holds every well monitored during the simulation
    """


    def __init__(self, path):
        """path (str): path to the observation file produced by load_results()"""
        f = open(path,'r')

        self.nbr_obs_points = int(f.readline().partition(';')[0])
        """nbr_obs_points (int): Number of observations points in the model, corresponding to the amount of measures done on the field"""
        self.ensemble_size = int(f.readline().partition(';')[0])
        """ensemble_size (int): Number of models in the ensemble"""
        self.wells = [[Well for j in range(self.ensemble_size)] for i in range(self.nbr_obs_points)]
        """wells (Well): List of Well for each observation point in each model if the ensemble"""

        for i in range(self.nbr_obs_points):
            name = f.readline().partition(';')[0]
            size = int(f.readline().partition(';')[0])
            x = float(f.readline().partition(';')[0])
            y = float(f.readline().partition(';')[0])
            z = float(f.readline().partition(';')[0])

            for k in range(size):
                temp = (f.readline()).split(',')
                temp_data = np.array(temp[1:])
                temp_time = datetime.strptime(temp[0],"%m/%d/%Y %H:%M:%S")

                for j in range(self.ensemble_size):
                    if k ==0 :
                        
                        self.wells[i][j] = Well(name, size, x, y, z)
                        self.wells[i][j].set_data_index(temp_data[j], temp_time,  k)
                    else :
                        self.wells[i][j].set_data_index(temp_data[j], temp_time,  k)

        f.close()


def load_results(assim_number, path_data):
    """
    Loading data in the folder path_data/ with the nth assim_number.
    Data are formated to be given to the assimilation process

    Initial time is hardcoded in the fucntion 

    /!\ Don't forget to modify it

    Args:
        assim_number (int): file path to your file containing all the observed data at each time step
        path_data (str): file path to the root file of the project
    
    Returns:
        None: Save all measures and observations in two text files in the path_data/Data/Assim_(assim_number)
    """
  
    initial_date = datetime(2021,9,1,9,0)

    # Relative = False indicate that results are stored as absolute values
    # Relative = True will store results relatively to the first data available at t = 0
    relative = False


    
    observationPoints = pd.read_csv(path_data+"Data/FEFLOW_Model/well_location.inp")


    obsPointNames = observationPoints.LocName.unique()
    numberOfWell = len(obsPointNames)
    
    coordinates = np.zeros((numberOfWell,3))
    
    for i in range(numberOfWell) :
        coordinates[i] = observationPoints[observationPoints.LocName==obsPointNames[i]].X,\
                        observationPoints[observationPoints.LocName==obsPointNames[i]].Y,\
                        observationPoints[observationPoints.LocName==obsPointNames[i]].Z
    
    referenceHeads = pd.read_csv(path_data+"Data\FEFLOW_Model\well_data.smp")
    referenceHeads = referenceHeads.dropna()

    # Writing the outputs file

    


    path = path_data+"Data/Assim_"+str(assim_number)+"/Head_resampled/"
    files = [f for f in listdir(path) if isfile(join(path, f))]


    tempHeads = np.zeros((len(files),732,numberOfWell+1))

    i = 0
    for file in files : 

        temp = np.load(path_data+"Data/Assim_"+str(assim_number)+"/Head_resampled/"+file)
        
        if relative :
            for j in range(tempHeads.shape[2]-1):
                temp[j+1] -= np.load(path_data+"/Data/Assim_"+str(assim_number)+"/Steady_head/steady_"+file[0:6]+".npy")[0][j]
        
        tempHeads[i] = temp.T
        i+=1


    f = open(path_data+"Data/Assim_"+str(assim_number)+"/measured_head.txt",'w')
    g = open(path_data+"Data/Assim_"+str(assim_number)+"/observed_head.txt",'w')

    f.write(str(numberOfWell)+" ; number of observation point\n")
    g.write(str(numberOfWell)+" ; number of observation point\n")
    g.write(str(tempHeads.shape[0])+" ; number of ensemble\n")


    for i in range(numberOfWell):

        times = referenceHeads[referenceHeads.LocName==obsPointNames[i]].ObsDate.values

        """To revert below if relative error
        """
        if relative : 
            heads = -referenceHeads[referenceHeads.LocName==obsPointNames[i]].ObsVal.values
        else :
            heads = -referenceHeads[referenceHeads.LocName==obsPointNames[i]].ObsVal.values + observationPoints[observationPoints.LocName==obsPointNames[i]].HHmasl.values

        time_delta = datetime.strptime(times[0],"%m/%d/%Y %H:%M:%S")-initial_date
        time_delta = int(time_delta.days*24 + time_delta.seconds//3600)


        num_to_skip = len(np.where(heads > 500)[0]) # No value set at 999

        f.write(str(obsPointNames[i])+" ; name of observation point\n")
        f.write(str(len(heads) - num_to_skip)+" ; number of measures\n")
        f.write(str(coordinates[i][0])+" ; x coordinate of observation point\n")
        f.write(str(coordinates[i][1])+" ; y coordinate of observation point\n")
        f.write(str(coordinates[i][2])+" ; z coordinate of observation point\n")
        

        g.write(str(obsPointNames[i])+" ; name of observation point\n")
        g.write(str(len(heads) - num_to_skip)+" ; number of measures\n")
        g.write(str(coordinates[i][0])+" ; x coordinate of observation point\n")
        g.write(str(coordinates[i][1])+" ; y coordinate of observation point\n")
        g.write(str(coordinates[i][2])+" ; z coordinate of observation point\n")


        for j in range(len(times)):
            if heads[j] < 500 : # No value set at 999
                f.write(str(times[j])+","+str(heads[j])+"\n")

                g.write(str(times[j])+",")
                
                for k in range(tempHeads.shape[0]-1):
                    if relative :
                        g.write(str(tempHeads[k][j+time_delta][i+1]-tempHeads[k][time_delta][i+1])+",") # i+1 because i=0 contain time
                    else :
                        g.write(str(tempHeads[k][j+time_delta][i+1])+",") # i+1 because i=0 contain time
                
                if (tempHeads.shape[0]==1):
                    k = 0
                else :
                    k+=1
                if relative :
                    g.write(str(tempHeads[k][j+time_delta][i+1]-tempHeads[k][time_delta][i+1])+"\n")
                else :
                    g.write(str(tempHeads[k][j+time_delta][i+1])+"\n")

    f.close()
    g.close()



def resample_time(path_data, path_save):

    """
    Take observations coming from FEFLOW and resampled them to a given time sampling
    Time sampling is hardcoded in the fucntion 

    /!\ Don't forget to modify it

    Args:
        path_data (str): file path to your file containing all the observed data at each time step
        path_save (str): path to the folder to store the resampled data
    
    Returns:
        None: Save all resampled files in the path_save folder
    """

    files = [f for f in listdir(path_data) if isfile(join(path_data, f))]

    for f in files : 

        data = np.load(path_data+f)


        resampled_data = np.zeros((732,26)) # Hardcoded, 732 time step in feflow, 27 well + 1 time column

            
        reinterp_hours = np.arange(0,732,dtype=float) # 732 hours in feflow run

        for j in range(25):
            resampled_data.T[j+1] = np.interp(reinterp_hours,data.T[0]*24-0.233,data.T[j+1]) # *24-0.33 to remove the 14 minutes 

        resampled_data.T[0] = reinterp_hours

        np.save(path_save+(f[0:-4]+'_resampled').split('/')[-1],resampled_data.T)

