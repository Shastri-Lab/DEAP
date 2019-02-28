import numpy as np

from helpers import getOutputShape


def _sanitizeInputs(input, kernel):
    assert 2 <= len(input.shape) <= 3
    if len(input.shape) == 2:
        input = np.expand_dims(input, 2)

    assert 2 <= len(kernel.shape) <= 4
    if len(kernel.shape) == 2:
        input = np.expand_dims(input, 2)
    if len(kernel.shape) == 3:
        input = np.expand_dims(input, 3)

    return input, kernel


def conv2d(input, kernel, padding, stride):
    """
    Input is a 3D matrix with index values row, col, depth, index
    Kernel is a 4D matrix with index values row, col, depth, index.
        The depth of the kernel must be equal to the depth of the input.
    """
    input, kernel = _sanitizeInputs(input, kernel)

    outputShape = getOutputShape(input.shape, kernel.shape, padding, stride)
    outputShape[2] = kernel.shape[3]
