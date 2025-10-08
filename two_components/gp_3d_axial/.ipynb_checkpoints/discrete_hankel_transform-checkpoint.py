# -*- coding: utf-8 -*-                                                  
"""                                                                      
Hankel transform module
geometry: 3D-axial
author: Federico Serrano
Physics and Astronomy Department
Washington State University
"""  
import numpy as np
from scipy.special import jv, jn_zeros

def dht(array, order=0, axis=-1):
    """
    Returns the discrete Hankel transform (not notmalized)

    Args:
        array: numpy array of general shape.
        order: order of the Bessel functions.
        axis: takes the given axis of array.
    """

    size = array.shape[axis]
    new_array = np.swapaxes(array, axis, -2)
    zeros = jn_zeros(order, size)
    factor = jn_zeros(order, size+1)[-1]

    grid = np.outer(zeros, zeros) / factor
    
    kernel = jv(order, grid)
    weight = np.diag(1/(jv(order+1, zeros)**2))
    hankel = kernel @ weight

    new_array = hankel @ new_array
    new_array = np.swapaxes(new_array, axis, -2)

    return (2/factor**2)*new_array

def idht(array, order=0, axis=-1):
    """                                                                         
    Returns the inverse discrete Hankel transform (not notmalized)

    Args:
        array: numpy array of general shape.
        order: order of the Bessel functions.
        axis: takes the given axis of array.
    """

    size = array.shape[axis]
    new_array = np.swapaxes(array, axis, -2)

    zeros = jn_zeros(order, size)
    factor = jn_zeros(order, size+1)[-1]

    grid = np.outer(zeros, zeros) / factor

    kernel = jv(order, grid)
    weight = np.diag(1/(jv(order+1, zeros)**2))
    hankel = kernel @ weight

    new_array = hankel @ new_array
    new_array = np.swapaxes(new_array, axis, -2)

    return 2*new_array
