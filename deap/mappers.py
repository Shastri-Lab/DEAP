import numpy as np

from deap.helpers import bisect_min, getOutputShape
from deap.photonics import MRRTransferFunction
from deap.photonics import MRMTransferFunction
from deap.photonics import PWB
from deap.photonics import LaserDiodeArray
from deap.photonics import ModulatorArray
from deap.photonics import PWBArray
from deap.photonics import PhotonicConvolver


class NeuronMapper:
    """
    Class that that maps a set of weights to an equivalent
    photonic pwb.
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
    # These will be populated when a pwb is created.
    _phase = None
    _dropput = None

    def setPrecision(newPrecision):
        assert 1 <= newPrecision
        if newPrecision % 2 == 0:
            newPrecision -= 1

        NeuronMapper._precision = newPrecision
        NeuronMapper._phase = None
        NeuronMapper._dropput = None

    def computePhaseShifts(weights):
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

        return phaseShifts, outputGain

    def build(weights):
        """
        Creates a new photonic pwb from a set of weights
        """
        phaseShifts, outputGain = \
            NeuronMapper.computePhaseShifts(weights)
        return PWB(phaseShifts, outputGain)

    def updateWeights(photonicNeuron, weights):
        """
        Updates an existing photonic pwb from a set of weights
        """
        phaseShifts, outputGain = \
            NeuronMapper.computePhaseShifts(weights)
        photonicNeuron._update(phaseShifts, outputGain)


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

    def computePhaseShifts(intenstiyMatrix, normval):
        assert not np.any(intenstiyMatrix < 0)
        if normval is None:
            normval = max(np.amax(intenstiyMatrix), 1)
        normalized = intenstiyMatrix / normval
        return ModulatorArrayMapper._mrm.phaseFromThroughput(
                    normalized)

    def build(inputs, normval=None):
        phaseShifts = \
            ModulatorArrayMapper.computePhaseShifts(inputs, normval)
        return ModulatorArray(phaseShifts)

    def updateInputs(modulatorArray, inputs, normval=None):
        phaseShifts = \
            ModulatorArrayMapper.computePhaseShifts(inputs, normval)
        modulatorArray._update(phaseShifts)


class PWBArrayMapper:
    """
    Class that maps a convolved matrix using photonic pwbs
    """

    def _createConnectionGraph(
            inputShape, kernel, stride, outputShape):
        connections = np.full(
                outputShape + (kernel.shape[0] * kernel.shape[1], 2),
                fill_value=-1)
        counts = np.zeros(inputShape)
        filterSize = kernel.shape[0]

        for row in range(connections.shape[0]):
            for col in range(connections.shape[1]):
                rowStart = row * stride
                colStart = col * stride
                colEnd = min(colStart + filterSize, inputShape[1])
                rowEnd = min(rowStart + filterSize, inputShape[0])
                R, C = np.mgrid[rowStart:rowEnd, colStart:colEnd]
                counts[rowStart:rowEnd, colStart:colEnd] += 1
                conn = np.column_stack((R.ravel(), C.ravel()))
                connections[row, col, :conn.shape[0], :] = conn

        return connections, counts

    def _setWeights(
            pwbs, outputShape, connections, kernel, sharedCounts, stride):
        for row in range(outputShape[0]):
            for col in range(outputShape[1]):
                conn = connections[row, col]
                # Get the number of times the  inupts were shared
                count = sharedCounts[conn[:, 0], conn[:, 1]].ravel()

                # Assign the weights using the kernel
                rDiff = row * stride
                cDiff = col * stride
                weights = count * \
                    kernel[conn[:, 0] - rDiff, conn[:, 1] - cDiff] \
                    .ravel()

                if pwbs[row, col] is None:
                    pwbs[row, col] = \
                        NeuronMapper.build(weights)
                else:
                    NeuronMapper.updateWeights(
                        pwbs[row, col], weights)
        return pwbs

    def updateKernel(pwbArray, newKernel):
        PWBArrayMapper._setWeights(
            pwbArray.pwbs,
            pwbArray.connections.shape[:2],
            pwbArray.connections,
            newKernel,
            pwbArray.sharedCounts,
            pwbArray.stride)

    def build(inputShape, kernel, stride=1):
        assert kernel.ndim == 2 or kernel.ndim == 3
        assert kernel.shape[0] == kernel.shape[1]

        outputShape = getOutputShape(inputShape, kernel.shape, 0, stride)[:-1]
        connections, sharedCounts = \
            PWBArrayMapper._createConnectionGraph(
                inputShape, kernel, stride, outputShape)

        pwbs = PWBArrayMapper. \
            _setWeights(
                np.full(outputShape, fill_value=None, dtype=object),
                outputShape, connections, kernel, sharedCounts, stride)

        return PWBArray(
                inputShape,
                connections,
                pwbs,
                sharedCounts,
                stride)


class PhotonicConvolverMapper:
    """
    Class that builds an entire photonic convolver that is capabale of
    performing a full convolution.
    """

    def build(image=None, kernel=None, stride=1, power=1,
              imageShape=None, kernelShape=None, normval=1):

        if image is None:
            assert imageShape is not None
            image = np.zeros(imageShape)

        if kernel is None:
            assert kernelShape is not None
            kernel = np.zeros(kernelShape)

        laserDiodeArray = LaserDiodeArrayMapper.build(
                kernel.shape, image.shape, power)
        modulatorArray = ModulatorArrayMapper.build(
                image, normval=normval)
        photonicNeuronArray = PWBArrayMapper.build(
                image.shape, kernel, stride)

        return PhotonicConvolver(
                laserDiodeArray,
                modulatorArray,
                photonicNeuronArray)
