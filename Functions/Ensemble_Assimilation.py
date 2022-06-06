__docformat__ = "google"

#-----------------------------------------------------------------------------------
# Reference:  P. Raanes
# https://github.com/nansencenter/DAPPER/blob/master/da_methods/da_methods.py
# Reference: Ceci Dip LIAMG
# https://github.com/groupeLIAMG/EnKF/blob/master/ENKF.py
# Reference: Evensen, Geir. (2009):
# "The ensemble Kalman filter for combined state and parameter estimation."
#-----------------------------------------------------------------------------------


#-----------------------------------------------------------------------------------
# Ensemble Kalman Filter class by Thomas Beraud, 2019
# INRS ETE Quebec
#-----------------------------------------------------------------------------------

#------- Import -------#

from random import gauss
from select import select
import numpy as np
import scipy as sp
from scipy.linalg import solve
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from numpy.random import multivariate_normal
from scipy.sparse import csr_matrix
import time



import numpy.linalg as nla


def mrdiv(b,A):
    """
    Solve a linear matrix equation, or system of linear scalar equations.
    Computes the "exact" solution, x, of the well-determined, i.e., full rank, linear matrix equation Ax = b.
    
    Args:
        b (float): values
        A (float): matrix like

    Returns:
        Solution to the system A x = b. Returned shape is identical to b.
    """
    return nla.solve(A.T,b.T).T


def rmse(predictions, targets):
    """
    Compute the RMSE between two arrays


    Args:
        predictions (float): array of predictions
        targets (float): array of targets

    Returns:
        RMSE (float)
    """	

    return np.sqrt(((predictions - targets) ** 2).mean())

def get_stats(x):
    """
    Compute statistics on an array

    Args:
        x (float): array of values

    Returns:
        median, p25, p75, STD of x
    """	
    return (np.quantile(x,0.5),np.quantile(x,0.25),np.quantile(x,0.75),np.std(x))


# Compute only the remaining rmse on the calibrated part of the data, the validation part will be unknown in a ideal case
# Only taking the 3 assimilated wells

def compute_worst_individual(measured,observations,min_,max_,num_to_drop,mode=True):
    """
    Compute the worst model in an ensemble of models

    Args:
        measured (Observation): Observation class holding every measured Well
        observations (Observation): Observation class holding every observed Well in the model
        min_ (float): Lower bound of data to keep to compute the RMSE. [0,1] < max_
        max_ (float): Higher bound of data to keep to compute the RMSE. [0,1] >= min_
        num_to_drop (int): Number of model to drop in the ensemble
        mode (bool): If True keeping the raw RMSE, if False keeping the RMSE scaled by max RMSE by Well
    Returns:
        index (int): Array of Index of the worst models in the ensemble, lenght of num_to_drop
    """

    ensemble_number = len(observations.wells[0])
    data = measured.wells[0].get_well_data()
    wells = len(measured)

    ref_puit = np.zeros((wells,int((max_*len(data[1])))-int(min_*len(data[1]))))

    for i in range(wells):
        ref_puit[i] = measured.wells[i].get_well_data()[1][int(min_*len(data[1])):int(max_*len(data[1]))]

    rmse_calibration = np.zeros((wells,ensemble_number))

    for j in range(wells):
        for i in range(ensemble_number):
            rmse_calibration[j][i] = rmse(observations.wells[j][i].get_well_data()[1][int(min_*len(data[1])):int(max_*len(data[1]))],ref_puit[j])


    weights = np.zeros(len(rmse_calibration[0]))

    if mode == True :
        for i in range(wells) :
                            
            for j in range(len(rmse_calibration[i])) :
                weights[j] += rmse_calibration[i][j]

    else :

        for i in range(wells) :

            for j in range(len(rmse_calibration[i])) :
                weights[j] += rmse_calibration[i][j]/np.max(ref_puit[i])

    return(np.argsort(weights)[-num_to_drop:])



def Measurements(  measurements_vector,
                   measurements_error_percent,
                   number_of_ensemble,
                   show_stats=False,
                   show_cov_matrix=False,
                  ):
        """
        Compute the covariance matrix and the measures with noise.

        Args:
            measurements_vector (float): Array of measures
            measurements_error_percent (float): Gaussian noise added to measures
            number_of_ensemble (int): Number of models in the ensemble
            show_stats (bool): If True print Measurement statistics (number of measures, mean, variance, std)
            show_cov_matrix (bool): If True, plot the Data matrix in the ensemble after the noise addition and plot the covariance matrix

        Returns:
            Cov (float): Covariance matrix between measures. Size number_of_ensemble * number_of_ensemble
            D (np.matrix): Created ensemble of measures. Size number of measures * number_of_ensemble
        """


        d = measurements_vector
        N = number_of_ensemble
        m = measurements_vector.shape[0]
        D = np.zeros((N,m))
        Cov = np.zeros((m,m))
        var_temp = np.zeros((m,N))


        for simu in range(N):
            for elt in range(len(d)):
                temp = gauss(d[elt],(measurements_error_percent))
                if temp >= 0 :
                    D[simu][elt] = temp
                else :
                    D[simu][elt] = 0


        D = np.matrix(D.T)
        mean_D = np.matrix(d).T

        anomalies = D - mean_D
        Cov = anomalies @ anomalies.T / (N - 1)


        if show_stats:
            print("#----- Measurements statistics -----#\n"+
            "Number of measures : %.0f" % d.shape[0] +"\n"+
            "Mean of measures : %.6f" % np.mean(d) +"\n"+
            "Variance of measures : %.6f" % np.var(d,dtype=np.float64) +"\n"+
            "Standard Deviation of measures : %.4f" % np.sqrt(np.var(d,dtype=np.float64)) +"\n"+
            "#----------------------------------#\n")

        if show_cov_matrix:

            plt.figure(figsize=(7,7))
            plt.imshow(D,aspect='auto')
            plt.colorbar()
            plt.title('Data Measurements Matrix after perturbation')
            plt.show()


            plt.figure(figsize=(7,7))
            plt.imshow(Cov)
            plt.title('Covariance Measurements Matrix')
            plt.colorbar()
            plt.show()



        return (Cov,D)



class Assimilation():

    
    """
    Assimilation Class holds the methods and structure to assimilate the data.

    These methods are based on the publications of Patrick Nima Raanes
    


     --------------   Importing Ensemble dataset   --------------

    The ensemble matrix should be formated as each row
    contained a parameter and each column a simulation

    If you run 100 simulations with 10 000 grid parameters
    to estimate, the ensemble matrix is of shape :

    (10 000, 100)
    

    --------------   Importing Observations Dataset   --------------

    The ensemble matrix should be formated as each row
    contained one observation and each column a simulation

    To use an Ensemble Smoother you should update the model by
    assimilating all the data in one run. So you have to stack
    all the observed data in one matrix

    If you run 100 simulations, with 20 observations parameters
    at each timestep and 50 timestep, the observation matrix is of shape :

    1000 observations = 20 * 50

    (1000, 100)

    Observation matrix :

    |   observations at 1 time-step  |
    |   observations at 2 time-step   |
    |               .                |
    |               .                |
    | observations at last time-step |
    


    --------------   Importing Observations Localisation   --------------


    Each observation have to be localized in the grid parameters we are
    evaluating. For each observation we have to had the number of the
    corresponding element in the grid.

    The localisation vector have to be the same length as the observation matrix.
    This is possible to look at different point at each time-step. To allow this
    flexibilty user have to defined the complete vector himself. If this is only
    the same point that are sampled in the gridat each time-step, the user can
    simply repeat the observations vector for the number of timestep.

    In background, a sparse observation matrix will be build to allow computation
    between the parameters matrix and only the sampled points.

    If you run 100 simulations, with 20 observations parameters
    at each timestep and 50 timestep, the localisation vector is of shape :

    (1000, 1)

    Observation matrix :

    |   localisation of observations at 1 time-step  |
    |   localisation of observations at 2 time-ste   |
    |               .                |
    |               .                |
    | localisation of observations at last time-step |
    """

    def __init__(self,
                   ensemble_matrix,
                   measurements_vector,
                   observations_vector,
                   measurements_error_percent,
                   inflation_factor = 1.01,
                   show_parameters=False,
                  ):
        """An Assimilation Class is initiated with all these parameters

        Args:
            ensemble_matrix (np.matrix): Matrix holding all parameters of the ensemble. Size is number of parameters * number of ensemble
            measurements_vector (np.matrix): Matrix holding all measures with added noise in the ensembe. Size is number of measures * number of ensemble
            observations_vector (np.matrix): Matrix holding all observations in the ensembe. Size is number of observations * number of ensemble
            measurements_error_percent (float): Noise to add on measures to create the ensemble of measures.
            inflation_factor = 1.01 (float): Inflation factor to reduce the variance collapse
            show_parameters = False (bool): If True display ensemble informations (number of ensembe, number of parameter, number of measures, std of observations). Will also display usefull informations for debugging purpose, as all intermediate matrix.
        """

        self.m = ensemble_matrix.shape[0]
        self.N = ensemble_matrix.shape[1]
        self.p = measurements_vector.shape[0]
        self.Ef = ensemble_matrix
        self.Ea = np.zeros((ensemble_matrix.shape))
        self.y = measurements_vector
        self.hE = np.zeros((self.p,self.N))
        self.measurements_error_percent = measurements_error_percent
        self.display = show_parameters
        self.Y = np.zeros((self.p, self.N))
        self.YC = np.zeros((self.p, self.N))
        self.inflation_factor = inflation_factor
        self.hE = observations_vector

        if show_parameters :
            print("#----- Ensemble Kalman Filter -----#\n"+
            "Number of ensemble : %.0f" % (self.N)+"\n"+
            "Number of parameter : %.0f" % (self.m)+"\n"+
            "Number of measurements : %.0f" % (self.p)+"\n"+
            "Standard Deviation of observations : %.6f" % np.sqrt(np.var(observations_vector,dtype=np.float64)) +"\n"+
            "#----------------------------------#\n")


    def pseudo_inverse(self):
        """Compute a pseudo inverse matrix of the covariance anomalies matrix

        Iterative method retaining the best pseudo inverse between five regularizationa and five TSVD.
        The number of iteration is hard coded in the function. 

        Args:
            None

        Returns:
            None: Store the pseudo_inverse in an attribute of Assimilation Class

        """

        # Computing covariance matrix of observations anomalies
        C = self.Y.T @ self.Y
        C = C / (self.N - 1) 
    
        # Initialize high error, aim is to find lower error at each iteration
        error = 100

        YC_Final = np.zeros((self.m, self.N))

        # Iterative double pseudo-inverse method to select the best pseudo-inverse
        for i in range(5):

            # Regularization method
            # Increase regularization factor of 1 order at each iteration
            C_regu  =  ( C + 1e-22*10**(i)*np.eye(self.N)/(self.N-1) )
            try :
                YC_regu = mrdiv(self.Y,C_regu)
                temp =  np.divide(YC_regu@C_regu - self.Y, self.Y, out=np.zeros_like(YC_regu@C_regu - self.Y), where=self.Y!=0)
                
                if np.max(temp) != 0 :
                    if ( abs(np.max(temp)) < error ):
                        error = abs( np.max(temp) )
                        YC_Final = YC_regu
            except:
                print("Error, inversion failed. Increasing regularization to "+str(1e-22*10**(i)))
                
            # TSVD Method
            # At each iteration we retain another singular value
            U,S,V = np.linalg.svd(C,full_matrices=True)
            rank = i*i+1
            pseudo_low_rank = V[:rank, :].T @ np.linalg.inv(np.diag(S[:rank])) @ U[:, :rank].T

            YC_svd = self.Y@pseudo_low_rank
            temp =  np.divide(YC_svd@C - self.Y, self.Y, out=np.zeros_like(YC_svd@C - self.Y), where=self.Y!=0)

            if np.max(temp) != 0 :                        
                if ( abs(np.max(temp)) < error ):
                    error = abs( np.max(temp) )
                    YC_Final = YC_svd

        self.YC = YC_Final 

        if self.display :

            print("Max error \"pseudo inverse\"",error)

            plt.figure(figsize=(7,7))
            plt.title("C")
            plt.imshow(C)
            plt.colorbar()
            plt.show()

            plt.figure(figsize=(10,2))
            plt.title("Reference Observation Anomalie")
            plt.imshow(self.Y,aspect='auto')
            plt.colorbar()
            plt.show()

            plt.figure(figsize=(10,2))
            plt.title("Pseudo Low Rank / Inverse")
            plt.imshow(self.YC,aspect='auto')
            plt.colorbar()
            plt.show()

            plt.figure(figsize=(10,2))
            plt.title("Y@Y-1")
            plt.imshow(self.YC@C,aspect='auto')
            plt.colorbar()
            plt.show()

            plt.figure(figsize=(10,2))
            plt.title("Erreur residuelle sur Initial")
            plt.imshow((self.YC@C-self.Y)/self.Y,aspect='auto')
            plt.colorbar()
            plt.show()





    def stochastic(self):
        """Compute the update on every parameters of the ensemble thanks to the assimilation process.

        The stochastic scheme is described by Patrick Nima Raanes.

        Args:
            None

        Returns:
            None: Store the updated parameters in an attribute of Assimilation Class, Ea

        """


        (self.Cd, self.D)  = Measurements(self.y,
                          self.measurements_error_percent,
                          self.N,
                          show_stats=self.display,
                          show_cov_matrix=self.display)


        mu = np.mean(self.Ef,1) # Computation of the ensemble mean

        A  = self.Ef - mu # Computation of the ensemble anomaly,
                           # individual deviation from the mean in each cell of each simulation

        hx = np.repeat(np.matrix(np.mean(self.hE,1)).T,self.N,axis=1) # Computation of the observed data mean

        self.Y  = self.hE- hx # Computation of the observed anomaly

        self.pseudo_inverse()

        temp_YC = self.YC / (np.max(np.abs(self.YC))*np.max(np.abs( self.D - self.hE )))
        # temp_YC is rescaled to allowed a maximal update equal to the maximum ensemble anomaly
        KG = A @ temp_YC.T

        if self.display :

            plt.figure(figsize=(10,2))
            plt.title("Parameters Anomalie")
            plt.imshow(A,aspect='auto')
            plt.colorbar()
            plt.show()

            plt.figure(figsize=(7,7))
            plt.imshow(KG,aspect='auto',cmap='RdYlGn')
            plt.colorbar()
            plt.title('Kalman Gain')
            plt.show()


        self.dE = self.inflation_factor * (KG @ ( self.D - self.hE ))

        # Scaling update to the parameters anomalies scale

        self.Ea   = self.Ef + self.dE

        if self.display :


            plt.figure(figsize=(7,7))
            plt.imshow(self.dE,aspect='auto',cmap='RdYlGn')
            plt.colorbar()
            plt.title('Update dE')
            plt.show()
            

    def deterministic(self):
        """Compute the update on every parameters of the ensemble thanks to the assimilation process.

        The deterministic scheme is described by Geir Evensen.

        Args:
            None

        Returns:
            None: Store the updated parameters in an attribute of Assimilation Class, Ea

        """
        
        S = self.hE - np.repeat(np.matrix(np.mean(self.hE,1)).T,self.N,axis=1)
        S_mean = np.mean(self.hE,axis=1)

        d = self.y

        E = np.zeros((self.N,self.p))
        for i in range(self.N):
            E[i] = d * (1 + self.measurements_error_percent*(np.random.rand(self.p)-0.5))

        E = E.T

        A = self.Ef

        # Parameters ensemble mean
        A_mean = np.repeat(np.mean(A,1),self.N,axis=1)
        A_prime = A - A_mean

        # S_0 and S_0_pseudo only hold the diagonal values
        U_0, S_0, V_0 = np.linalg.svd(S) 

        S_0_pseudo = 1 / S_0
        # Evensen 2004, subspace is defined by the first N-1 singular values
        S_0_pseudo[-1] = 0

        temp = np.zeros((V_0.shape[0], U_0.shape[0]))
        temp[:V_0.shape[0], :V_0.shape[0]] = np.diag(S_0_pseudo)
        S_0_pseudo = temp

        temp1 = np.zeros((U_0.shape[0], V_0.shape[0]))
        temp1[:V_0.shape[0], :V_0.shape[0]] = np.diag(S_0)
        S_0 = temp1

        
        X_0 = S_0_pseudo@U_0.T@E

        U_1, S_1, V_1 = np.linalg.svd(X_0) 
        temp2 = np.zeros((U_1.shape[0], V_1.shape[0]))
        temp2[:V_1.shape[0], :V_1.shape[0]] = np.diag(S_1)
        S_1 = temp2
        
        X_1 = U_0@S_0_pseudo.T@U_1

        # Python should return N,1 but return 1,N  .T to solve that
        y_0 = (X_1.T@(d-S_mean)).T

        y_2 = np.linalg.inv(np.identity(self.N)+S_1**2)@y_0
        y_3 = X_1@y_2
        y_4 = S.T@y_3
        A_update = A_mean + (A-A_prime)@y_4

        X_2 = np.sqrt(np.linalg.inv((np.identity(self.N)+S_1**2)))@X_1.T@S

        U_2, S_2, V_2 = np.linalg.svd(X_2)
        temp3 = np.zeros((U_2.shape[0], V_2.shape[0]))
        temp3[:V_2.shape[0], :V_2.shape[0]] = np.diag(S_2)
        S_2 = temp3

        u,s,theta = np.linalg.svd(np.random.rand(self.N,self.N))

        self.Ea = A_update + A_prime@V_2@np.sqrt(np.identity(self.N)*self.inflation_factor-S_2.T@S_2)@(theta.T)

        self.dE = self.Ea - self.Ef

        if self.display :


            plt.title("Observation Anomalies")
            plt.imshow(S,aspect='auto')
            plt.colorbar()
            plt.show()


            plt.title("Ensemble Anomalies")
            plt.imshow(A_prime,aspect='auto')
            plt.colorbar()
            plt.show()

            plt.title("Mean update of ensemble")
            plt.imshow(A_update,aspect='auto')
            plt.colorbar()
            plt.show()

            plt.title("Final update of ensemble")
            plt.imshow(self.Ea,aspect='auto')
            plt.colorbar()
            plt.show()