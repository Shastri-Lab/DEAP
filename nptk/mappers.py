import numpy as np

from nptk.helpers import bisect_min, getOutputShape
from nptk.photonics import MRRTransferFunction
from nptk.photonics import MRMTransferFunction
from nptk.photonics import PhotonicNeuron
from nptk.photonics import LaserDiodeArray
from nptk.photonics import ModulatorArray
from nptk.photonics import PhotonicNeuronArray


class NeuronMapper:
    """
    Class that that maps a set of weights to an equivalent
    photonic neuron.
    """
    def _precomputeDropputToPhaseMapping(precision):
        """
        Creates a mapping between phase and dropput by index. For a given
        precision.

        I.e. At a an index i, a phase shift of phi[i] will correspond to a
        of dropput[i] and vice-versa.
        """
        mrr = MRRTransferFunction()
        dropput = np.linspace(1.1e-4, 1, precision)
        phase = mrr.phaseFromDropput(dropput)
        return phase, dropput

    _precision = 127

    # Precomputed phase and dropput values for a given precision.
    # These will be populated when a neuron is created.
    _phase = None
    _dropput = None

    def setPrecision(newPrecision):
        assert 1 <= newPrecision
        if newPrecision % 2 == 0:
            newPrecision -= 1

        NeuronMapper._precision = newPrecision
        NeuronMapper._phase = None
        NeuronMapper._dropput = None

    def build(weights):
        if NeuronMapper._phase is None or NeuronMapper._dropput is None:
            # Compute phase and dropput mapping if it hasn't been done
            # already.
            NeuronMapper._phase, NeuronMapper._dropput = \
                NeuronMapper._precomputeDropputToPhaseMapping(
                        NeuronMapper._precision)

        weights = np.asarray(weights)
        maxElement = np.amax(np.abs(weights))

        # Normalize the weights and compute the post-optical gain.
        outputGain = max(maxElement, 1)
        normalized_weights = weights / outputGain

        # Determine what dropput values are needed using the formula:
        # Td - Tp = w => Td - (1 - Td) = w => Td = (w + 1) / 2
        desiredDropputs = (normalized_weights + 1) / 2

        phaseShifts = np.zeros(weights.size)
        for i, desiredDropput in enumerate(desiredDropputs):
            index = bisect_min(NeuronMapper._dropput, desiredDropput)
            phaseShifts[i] = NeuronMapper._phase[index]

        return PhotonicNeuron(phaseShifts, outputGain)


class LaserDiodeArrayMapper:
    """
    Class that maps a convolution size and input size to an array of laser
    diodes.
    """
    def build(inputShape, outputShape, power=1):
        # Create adjacency list for input and output shape.
        grid = np.indices(outputShape)
        rows = grid[0] % inputShape[0]
        cols = grid[1] % inputShape[1]
        connections = np.dstack((rows, cols))

        return LaserDiodeArray(inputShape, outputShape, connections, power)


class ModulatorArrayMapper:
    """
    Class that maps a relative intenstiy matrix to an array of optical
    modulators.
    """
    _mrm = MRMTransferFunction()

    def build(intenstiyMatrix):
        assert not np.any(intenstiyMatrix < 0)

        normalized = intenstiyMatrix / max(np.amax(intenstiyMatrix), 1)
        phaseShifts = ModulatorArrayMapper._mrm.phaseFromThroughput(
                normalized)
        return ModulatorArray(phaseShifts)


class PhotonicNeuronArrayMapper:
    """
    Class that maps a convolved matrix using photonic neurons
    """

    def _createConnectionGraph(
            inputShape, kernel, padding, stride, outputShape):
        connections = np.full(outputShape, fill_value=None, dtype=object)
        neuralWeights = np.full(outputShape, fill_value=None, dtype=object)
        filterSize = kernel.shape[0]

        for row in range(connections.shape[0]):
            for col in range(connections.shape[1]):
                inputRow = row * stride
                inputCol = col * stride

                rowStart = max(0, inputRow - padding)
                colStart = max(0, inputCol - padding)
                colEnd = min(inputCol + filterSize - padding, inputShape[1])
                rowEnd = min(inputRow + filterSize - padding, inputShape[0])

                neuralWeights[row, col] = \
                    kernel[rowStart-inputRow+padding:rowEnd-inputRow+padding,
                           colStart-inputCol+padding:colEnd-inputCol+padding] \
                    .ravel()

                R, C = np.mgrid[rowStart:rowEnd, colStart:colEnd]
                connections[row, col] = \
                    np.column_stack((R.ravel(), C.ravel()))

        return connections, neuralWeights

    def build(inputShape, kernel, padding=0, stride=1):
        outputShape = getOutputShape(inputShape, kernel.shape, padding, stride)
        connections, neuralWeights = \
            PhotonicNeuronArrayMapper._createConnectionGraph(
                inputShape, kernel, padding, stride, outputShape)

        neurons = np.full(outputShape, fill_value=None, dtype=object)
        for row in range(outputShape[0]):
            for col in range(outputShape[1]):
                neurons[row, col] = NeuronMapper.build(neuralWeights[row, col])

        return PhotonicNeuronArray(inputShape, connections, neurons)
