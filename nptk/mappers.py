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
        if maxElement > 1:
            outputGain = maxElement
        else:
            outputGain = 1

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
        connections = np.full(inputShape, fill_value=None, dtype=object)
        for row in range(outputShape[0]):
            for col in range(outputShape[1]):
                connRow = row % inputShape[0]
                connCol = col % inputShape[1]
                if connections[connRow, connCol] is None:
                    connections[connRow, connCol] = set()
                connections[connRow, connCol].add((row, col))

        return LaserDiodeArray(inputShape, outputShape, connections, power)


class ModulatorArrayMapper:
    """
    Class that maps a relative intenstiy matrix to an array of optical
    modulators.
    """
    _mrm = MRMTransferFunction()

    def build(intenstiyMatrix):
        assert not np.any(intenstiyMatrix > 1)
        phaseShifts = ModulatorArrayMapper._mrm.phaseFromThroughput(
                intenstiyMatrix)
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

        paddedRows = inputShape[0] + 2 * padding
        paddedColumns = inputShape[1] + 2 * padding
        for row in range(0, paddedRows - filterSize + 1, stride):
            for col in range(0, paddedColumns - filterSize + 1, stride):
                # Corresponding output row
                outputRow = row // stride
                # Corresponding output column
                outputCol = col // stride

                # Iterate over indices that connect to a particular neuron
                rowStart = max(padding, row)
                colStart = max(padding, col)
                rowEnd = min(row + filterSize, paddedRows - padding)
                colEnd = min(col + filterSize, paddedColumns - padding)

                # Get required neural weights, exlcuding those that
                # touch padded values.
                neuralWeights[outputRow, outputCol] = \
                    kernel[rowStart-row:rowEnd-row,
                           colStart-col:colEnd-col].ravel()

                # Generate connections
                R, C = np.mgrid[rowStart-padding:rowEnd-padding,
                                colStart-padding:colEnd-padding]
                connections[outputRow, outputCol] = \
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
