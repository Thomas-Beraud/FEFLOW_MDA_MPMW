__docformat__ = "google"

import sys
import Feflow_Utils as FU   

"""
Defining all the paths needed to launch Feflow. To avoid error in simulation,
we entered absolute path to files.

/ ! \ Don't forget to modify the path of Feflow bin64 in the Feflow_Utils.py file, before importing
ifm38 module
"""

if __name__ == "__main__":

    # Data path to your Data folder containing kzone, element and recharge zone coming from Feflow
    path_data = "D:/Dropbox/FEFLOW_MDA_MPMW/Data/"
    fem_steady_model = path_data + "FEFLOW_Model/steady.fem"
    fem_transient_model = path_data + "FEFLOW_Model/transient.fem"

    # Script will be launched by a .bat script to avoid a memory leak iterating several Feflow models
    # parameters_from_bat are storing the model actually be done and the assimilation 
    parameters_from_bat = sys.argv

    assim_number = int(parameters_from_bat[1])
    model_number = int(parameters_from_bat[2])

    """
    In case you need to launch it without batch, all errors detection in bacth are
    coded in this function : 
    
    FU.check_launch_feflow(path_data,
                        fem_steady_model,
                        fem_transient_model,
                        assim_number,
                        model_number)
    """

    FU.launch_feflow(path_data,
                        fem_steady_model,
                        fem_transient_model,
                        assim_number,
                        model_number)