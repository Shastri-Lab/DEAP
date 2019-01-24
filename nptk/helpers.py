import numpy as np


def bisect_min(array, val):
    """
    Given a sorted array, return the index of an existing element
    that has the closest absolute distance to val.
    """

    array = np.asarray(array)
    index = np.searchsorted(array, val, side='left')

    if index == array.size:
        return array.size - 1
    elif index == 0:
        return 0

    if index == array.size - 1:
        offsetArray = np.array([
            array[index - 1],
            array[index]])
    else:
        offsetArray = np.array([
            array[index - 1],
            array[index],
            array[index + 1]])

    return np.argmin(np.abs(val - offsetArray)) - 1 + index
