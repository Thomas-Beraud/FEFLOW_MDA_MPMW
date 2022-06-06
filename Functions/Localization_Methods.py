__docformat__ = "google"
# build doc : pdoc D:\Dropbox\Run4_transitoire\Functions\Feflow_Utils
# pdoc -o D:\Dropbox\Run4_transitoire\Documentation D:\Dropbox\Run4_transitoire\Functions


"""
    This module holds the code to create a weighting localization around each observation point
"""

from typing import Union
import numpy as np
import pandas as pd



def get_ellipse_bb(major : float, 
                    minor : float, 
                    angle_rad : float) -> Union[float, float, float, float]:
    
    """
    Compute tight bounding box around ellipse of range major / minor


    Args:
        major (float): ellipse range along long axis, positive value
        minor (float): ellipse range along short axis, positive value
        angle_rad (float): radian trigonometrical rotation, 0 along x, increase counter clockwise

    Returns:
        Union[float, float, float, float]: x_min, y_min, x_max, y_max
    """	

    xp = np.cos(angle_rad) * major
    yp = np.sin(angle_rad) * major
    xq = np.cos(angle_rad - np.pi/2) * minor
    yq = np.sin(angle_rad - np.pi/2) * minor


    return np.floor(-np.sqrt(xp**2 + xq**2)), np.floor(-np.sqrt(yp**2 + yq**2)),\
				np.ceil(np.sqrt(xp**2 + xq**2)), np.ceil(np.sqrt(yp**2 + yq**2))



def validity_variogram(s : float, 
            n : float, 
            r : float) -> None:
    """
    Compute numerical validity of variogram model. Return an error if any fails

    Args:
        s (float): sill, positive value
        n (float): nugget, positive value
        r (float): range, positive value

    Raises:
        ValueError: Nugget effect, sill or range is negative, impossible
        ValueError: Nugget effect should be lower than sill
    """

    if (n < 0 or s < 0 or r < 0):
        raise ValueError("Nugget effect, sill or range is negative, impossible")

    if (n > s):
        raise ValueError("Nugget effect should be lower than sill")



def gaussian(s : float, 
            n : float, 
            r : float, 
            h : float) -> float:
    """
    Compute variance on a gaussian variogram model given the sill, range, nugget and distance of observation.   
    Formula defined by Chiles and Delfiner 1999.

    Args:
        s (float): sill, positive value
        n (float): nugget, positive value
        r (float): range, positive value
        h (float): distance, could be single value or a vector of several point to compute) 
    
    Returns:
        float: variance, single value or vector of values in function of the type of h
    """

    validity_variogram(s, n, r)

    return (s-n)*(1-np.exp((-h**2)/(r**2*(1/3)))) + n



def exponential(s : float, 
            n : float, 
            r : float, 
            h : float) -> float:
    """
    Compute variance on an exponential variogram model given the sill, range, nugget and distance of observation.   
    Formula defined by Chiles and Delfiner 1999.

    Args:
        s (float): sill, positive value
        n (float): nugget, positive value
        r (float): range, positive value
        h (float): distance, could be single value or a vector of several point to compute) 
    
    Returns:
        float: variance, single value or vector of values in function of the type of h
    """

    validity_variogram(s, n, r)

    return (s - n)*(1 - np.exp(-abs(h) / (r/3) )) + n



def spherical(s : float, 
            n : float, 
            r : float, 
            h : float) -> float:

    """
    Compute variance on a spherical variogram model given the sill, range, nugget and distance of observation.   
    Formula defined by Chiles and Delfiner 1999.

    Args:
        s (float): sill, positive value
        n (float): nugget, positive value
        r (float): range, positive value
        h (float): distance, could be single value or a vector of several point to compute) 
    
    Returns:
        float: variance, single value or vector of values in function of the type of h
    """

    validity_variogram(s, n, r)

    temp = np.zeros(len(h))

    for i in range(len(h)):
        if h[i] < r : 
            temp[i] = (s - n) * ( ( (3 * abs(h[i])) / (2*r) ) - (abs(h[i])**3) / (2 * (abs(r)**3) ) ) + n
        else :
            temp[i] = s

    return temp



def get_elements_ellipsoid(data: pd.DataFrame, 
            max_range : float, 
            med_range : float, 
            min_range : float, 
            dip : float, 
            azimuth : float, 
            x_c : float, 
            y_c : float, 
            z_c : float, 
            sill : float, 
            nugget : float, 
            model : str) -> Union[int,float]:

    """
    Computes an observation weight for each point in a ellispoid around
    a given center and ellipse geometry. Weight equal at 1 on observation point, decaying
    following the given model, to the nugget on the ellispe boundary and outside. 
    
    It is used has localization method to improve numerical stability in further inversion and 
    reduces spurrious correlations. Take 3 vectors x, y, z containing coordinate of Feflow elements. 
    Combined with ellipse geometry, it returned index of element inside the ellipse and the weight 
    associated with every Feflow element.

    Numerical implementation required to pass by a local coordinate system centered on observation
    point, and rotation along dip and azimuth to use ellispoid axis as system axis.

    Args:
        data (pd.DataFrame): containing the coordinates X,Y,Z and the ElementId as 
        columns of Dataframe.
        max_range (float): maximum range of observation weight along the ellipsoid axis
        med_range (float): medium range of observation weight along the ellipsoid axis
        min_range (float): minimum range of observation weight along the ellipsoid axis
        dip (float): Angle in radian of max_range on an vertical plane, with 0 correpsonding to no dip,
            ranging from 0 to -pi/2, negative value to the center of earth
        azimuth (float): Angle in radian of max_range on an horizontal plane, with 0 pointing to north,
            ranging from 0 to 2pi clockwise
        x_c (float): x cell coordinate center of the observed cell
        y_c (float): y cell coordinate center of the observed cell
        z_c (float): z cell coordinate center of the observed cell
        sill (float): Equivalent to the sill in variogram, giving a weight to points around observation 
            point
        nugget (float): Equivalent to the nugget in variogram, giving a random noise weight to points around 
            observation point
        model (str): Variogram model type used to compute the weight decay in the ellipsoid. "gaussian", 
            "exponential" or "spherical". Models defined by Chiles and Delfiner 1999.

    Returns:
        index (int): Indexes of Feflow element inside the research ellispoid, 
            values (float): Observation weight associated to elements in index
    """

    x = data.X.values
    y = data.Y.values
    z = data.Z.values

    # Centering every Feflow elements around the observation point
    coords_local = np.matrix([(x - x_c),(y - y_c),(z - z_c)])

    # Rotation_along_y to remove dip effect
    mat = np.matrix([[np.cos(dip), 0, np.sin(dip)],
         [0, 1, 0],
         [-np.sin(dip), 0, np.cos(dip)]])

    coords_local = np.matmul(mat,coords_local)


    # Rotation_along_z to remove azimuth effect
    mat_ = np.matrix([[np.cos(-azimuth), -np.sin(-azimuth),0],
     [np.sin(-azimuth), np.cos(-azimuth), 0],
     [0, 0, 1]])

    coords_local = np.matmul(mat_,coords_local)

    # Transformation in a np.array to insure working element acces
    coords_local = np.array(coords_local)

    # Elements contained in a given ellispoid will respect a^2 + b^2 + c^2 <= 1
    a = np.square(coords_local[0]/max_range)
    b = np.square(coords_local[1]/med_range)
    c = np.square(coords_local[2]/min_range)

    temp = a+b+c 

    # Retaining only elements contained in ellipsoid, with a trick to allow a point
    # Localized on the observation point, numerically impossible (division by 0)
    values = np.zeros(len(np.where(temp == 0)[0]) + len(np.where(temp <= 1)[0]))
    index = np.zeros(len(np.where(temp == 0)[0]) + len(np.where(temp <= 1)[0]))

    values[0:len(np.where(temp == 0)[0])] = sill - nugget
    index[0:len(np.where(temp == 0)[0])] = np.where(temp == 0)[0]

    # Weighting computed as asked by the final user with respective model
    if model == "gaussian":

        values[-len(np.where(temp <= 1)[0]):] = (sill - (a[np.where(temp <= 1)[0]] * gaussian(sill,nugget,max_range,coords_local[0][np.where(temp <= 1)[0]]) + 
                     b[np.where(temp <= 1)[0]] * gaussian(sill,nugget,med_range,coords_local[1][np.where(temp <= 1)[0]]) +
                     c[np.where(temp <= 1)[0]] * gaussian(sill,nugget,min_range,coords_local[2][np.where(temp <= 1)[0]])))
        index[-len(np.where(temp <= 1)[0]):] = np.where(temp <= 1)[0]

    if model == "spherical":

        values[-len(np.where(temp <= 1)[0]):] = (sill - (a[np.where(temp <= 1)[0]] * spherical(sill,nugget,max_range,coords_local[0][np.where(temp <= 1)[0]]) +
                     b[np.where(temp <= 1)[0]] * spherical(sill,nugget,med_range,coords_local[1][np.where(temp <= 1)[0]]) +
                     c[np.where(temp <= 1)[0]] * spherical(sill,nugget,min_range,coords_local[2][np.where(temp <= 1)[0]])))
        index[-len(np.where(temp <= 1)[0]):] = np.where(temp <= 1)[0]

    if model == "exponential":

        values[-len(np.where(temp <= 1)[0]):] = (sill - (a[np.where(temp <= 1)[0]] * exponential(sill,nugget,max_range,coords_local[0][np.where(temp <= 1)[0]]) +
                     b[np.where(temp <= 1)[0]] * exponential(sill,nugget,med_range,coords_local[1][np.where(temp <= 1)[0]]) +
                     c[np.where(temp <= 1)[0]] * exponential(sill,nugget,min_range,coords_local[2][np.where(temp <= 1)[0]])))
        index[-len(np.where(temp <= 1)[0]):] = np.where(temp <= 1)[0]


    index = index.astype(int)

    # Indexs of elements in Feflow, weight associated with each element.

    return data.iloc[index,0].values, values


def extract_feflow_from_ellipsoid(data: pd.DataFrame,
                                  x_obs: float,
                                  y_obs: float,
                                  z_obs: float,
                                  max_range: float,
                                  med_range: float,
                                  min_range: float,
                                  azimuth: float,
                                  dip: float,
                                  sill: float,
                                  nugget: float,
                                  model: str) -> Union[int,float]:
                            
    """
    Computes an observation weight for each elements in a Feflow model. Weight is equal to 1 
    on observation point, decaying following the range and type of  variogram model. Outside 
    of the range, the weight is equal to the nugget. 
    
    It is used has localization method to improve numerical stability in further inversion and 
    reduces spurrious correlations. Take 4 vectors x, y, z and elementID of Feflow elements. 
    Combined with variogram geomtry, it returned index of elements inside the range and the weight 
    associated with every Feflow element.

    Args:
        data (pd.DataFrame): containing reindex from 0 to n_element, the coordinates X,Y,Z and 
        the ElementId as columns of Dataframe. Columns name : index, X, Y, Z, ElementId
        x_obs (float): x coordinate of observed point in Feflow model reference coordinate
        y_obs (float): y coordinate of observed point in Feflow model reference coordinate
        z_obs (float): z coordinate of observed point in Feflow model reference coordinate
        max_range (float): maximum range of observation weight along the ellipsoid axis
        med_range (float): medium range of observation weight along the ellipsoid axis
        min_range (float): minimum range of observation weight along the ellipsoid axis
        azimuth (float): Angle in degree of max_range on an horizontal plane with the north,
            0 degree is a pure northern azimut. Ranging from 0 to 360 degrees clockwise
            / ! \ Degrees increases opposite to trigonometric way. Real world map convention
        dip (float): Angle in degree of max_range on an vertical plane, with 0 correpsonding to no dip,
            ranging from 0 to -90, -90 is pointing to the earth center
        sill (float): Equivalent to the sill in variogram, giving a weight to points around observation 
            point
        nugget (float): Equivalent to the nugget in variogram, giving a random noise weight to points around 
            observation point
        model (str): Variogram model type used to compute the weight decay in the ellipsoid. "gaussian", 
            "exponential" or "spherical". Models defined by Chiles and Delfiner 1999.

    Raises:
        ValueError: Azimuth should be in radian between 0 and 2 pi
        ValueError: Dip should be in radian between 0 and -pi/2
        ValueError: Wrong variogram model, should be gaussian, spherical or exponential
        ValueError: Dataframe coordinates columns should be named exactly X,Y,Z

    Returns:
        index (int): Indexes of Feflow element inside the research ellispoid, 
            values (float): Observation weight associated to elements in index

    """

    if (azimuth < 0 or azimuth >= 360):
        raise ValueError("Azimuth should be in degree between 0 and 359.9")

    if (dip > 0 or dip < -90):
        raise ValueError("Dip should be in degree between 0 and -90")


    # Converts degree angle to radian angle, as degree are more commonly used in field measure
    azimuth = azimuth * np.pi/180
    dip = dip * np.pi/180

    if not (model == "gaussian" or model == "spherical" or model == "exponential"):
        raise ValueError("Wrong variogram model, should be gaussian, spherical or exponential")   

    if not np.all(data.columns[0:5]==["index","X","Y","Z","ElementID"]):
        raise ValueError("Dataframe coordinates columns should be named exactly index, X, Y, Z, ElementID")   

    # Transforming azimuth with 0 to the north to the classical trignonometric 0 on the east axis
    azimuth = -azimuth - np.pi/2

    
    #SpeedUp by doing a smaller subset of element in a bounding box.
    #Factor ranging from 0.9 to 15 as the ellipsoid search reduces around observation point
    x_min, y_min, x_max, y_max = get_ellipse_bb(max_range, med_range, azimuth)
    _, z_min, _, z_max = get_ellipse_bb(med_range, min_range, dip)
 
    data = data[np.logical_and(data.X.values > x_min + x_obs, data.X.values < x_max + x_obs)]
    data = data[np.logical_and(data.Y.values > y_min + y_obs, data.Y.values < y_max + y_obs)]
    data = data[np.logical_and(data.Z.values > z_min + z_obs, data.Z.values < z_max + z_obs)]
    

    index,values = get_elements_ellipsoid(data,
                                    max_range,
                                    med_range,
                                    min_range,
                                    dip,
                                    azimuth,
                                    x_obs,
                                    y_obs,
                                    z_obs,
                                    sill,
                                    nugget,
                                    model)

    return index, values



