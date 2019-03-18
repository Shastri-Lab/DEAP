import numpy as np

from deap.helpers import getOutputShape
from deap.mappers import PhotonicConvolverMapper
from deap.mappers import ModulatorArrayMapper
from deap.mappers import PWBArrayMapper


def convDEAP(image, kernel, stride):
    """
    Image is a 3D matrix with index values row, col, depth, index
    Kernel is a 4D matrix with index values row, col, depth, index.
        The depth of the kernel must be equal to the depth of the input.
    """
    assert image.shape[2] == kernel.shape[2]

    # Allocate memory for storing result of convolution
    outputShape = getOutputShape(image.shape, kernel.shape, stride=stride)
    output = np.zeros(outputShape)

    # Build the photonic circuit
    weightBanks = []
    inputShape = (kernel.shape[0], kernel.shape[1])
    for k in range(image.shape[2]):
        pc = PhotonicConvolverMapper.build(
                imageShape=inputShape,
                kernelShape=inputShape,
                power=255, normval=255)
        weightBanks.append(pc)

    for k in range(kernel.shape[3]):
        # Load weights
        weights = kernel[:, :, :, k]
        for c in range(weights.shape[2]):
            PWBArrayMapper.updateKernel(
                weightBanks[c].pwbArray,
                weights[:, :, c])

        for h in range(0, outputShape[0], stride):
            for w in range(0, outputShape[1], stride):
                # Load inputs
                inputs = \
                    image[h:min(h + kernel.shape[0], image.shape[0]),
                          w:min(w + kernel.shape[0], image.shape[1]), :]

                for c in range(kernel.shape[2]):
                    ModulatorArrayMapper.updateInputs(
                            weightBanks[c].modulatorArray, inputs[:, :, c],
                            normval=255)
                    output[h, w, k] += weightBanks[c].step()
    return output


def convDEAP_GIP(image, kernel, stride, convolverShape=None):
    """
    Image is a 3D matrix with index values row, col, depth, index
    Kernel is a 4D matrix with index values row, col, depth, index.
        The depth of the kernel must be equal to the depth of the input.
    """
    assert image.shape[2] == kernel.shape[2]
    assert kernel.shape[2] == 1 and kernel.shape[3] == 1
    if convolverShape is None:
        convolverShape = image.shape

    # Define convolutional parameters
    Hm, Wm = convolverShape[0], convolverShape[1]
    H, W = image.shape[0], image.shape[1]
    R = kernel.shape[0]

    # Allocate memory for storing result of convolution
    outputShape = getOutputShape(image.shape, kernel.shape, stride=stride)
    output = np.zeros(outputShape)

    # Load weights
    pc = PhotonicConvolverMapper.build(
            imageShape=convolverShape,
            kernel=kernel[:, :, 0, 0], power=255)

    input_buffer = np.zeros(convolverShape)
    for h in range(0, H - R + 1, Hm - R + 1):
        for w in range(0, W - R + 1, Wm - R + 1):
            inputs = image[h:min(h + Hm, H), w:min(w + Wm, W), 0]
            # Load inputs into a buffer if convolution shape doesn't tile
            # nicely.
            input_buffer[:inputs.shape[0], :inputs.shape[1]] = inputs
            input_buffer[inputs.shape[0]:, inputs.shape[1]:] = 0

            # Update the inputs to the system.
            ModulatorArrayMapper.updateInputs(
                pc.modulatorArray,
                input_buffer,
                normval=255)

            # Perform the convolution and store to memory
            result = pc.step()[:min(h + Hm, H) - h - R + 1,
                               :min(w + Wm, W) - w - R + 1]
            output[h:min(h + Hm, H) - R + 1,
                   w:min(w + Hm, W) - R + 1,
                   0] = result

    return output
