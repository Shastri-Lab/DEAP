import numpy as np
import matplotlib.pyplot as plt
from nptk.helpers import bisect_min


class MRRTransferFunction:
    """
    Computes the transfer function of a microring resonator (MMR).
    """
    def __init__(self, a=1, r1=0.99, r2=0.99):
        self.a = a
        self.r1 = r1
        self.r2 = r2
        self._minDropput = self.dropput(np.pi)

    def throughput(self, phi):
        """
        Calculates throuput of a MMR
        """
        num = (self.r2 * self.a)**2 - 2 * self.r1 * self.r2 * self.a * np.cos(phi) + self.r1**2 # noqa
        denom = 1 - 2 * self.r1 * self.r2 * self.a * np.cos(phi) + (self.r1 * self.r2 * self.a)**2  # noqa

        return num / denom

    def dropput(self, phi):
        """
        Calculates dropput of a MMR
        """
        num = (1 - self.r1**2) * (1 - self.r2**2) * self.a
        denom = 1 - 2 * self.r1 * self.r2 * self.a * np.cos(phi) + (self.r1 * self.r2 * self.a)**2  # noqa

        return num / denom

    def phaseFromDropput(self, Td):
        """
        Given a dropout, create a phase.
        """
        Td = np.asarray(Td)

        # Create empty array to store results
        ans = np.empty_like(Td)

        # For tiny dropput values, set to pi
        lessThanMin = Td <= self._minDropput
        minOrMore = ~lessThanMin
        ans[lessThanMin] = np.pi

        # For remaning, actually try to solve.
        num = ((1 - self.r1**2) * (1 - self.r2**2) * self.a / Td[minOrMore] - 1 - (self.r1 * self.r2 * self.a)**2) # noqa
        denom = -2 * self.r1 * self.r2 * self.a
        ans[minOrMore] = np.arccos(num / denom)

        return ans


class MRMTransferFunction:
    """
    Computes the transfer function of a microring modulator (MRM).
    """
    def __init__(self, a=0.9, r=0.9):
        self.a = a
        self.r = r
        self._maxThroughput = self.throughput(np.pi)

    def throughput(self, phi):
        I_pass = self.a**2 - 2 * self.r * self.a * np.cos(phi) + self.r**2
        I_input = 1 - 2 * self.a * self.r * np.cos(phi) + (self.r * self.a)**2
        return I_pass / I_input

    def phaseFromThroughput(self, Tn):
        Tn = np.asarray(Tn)

        # Create variable to store results
        ans = np.empty_like(Tn)

        # For high throuputs, set to pi
        moreThanMax = Tn >= self._maxThroughput
        maxOrLess = ~moreThanMax
        ans[moreThanMax] = np.pi

        # Now solve the remainng
        cos_phi = Tn[maxOrLess] * (1 + (self.r * self.a)**2) - self.a**2 - self.r**2  # noqa
        ans[maxOrLess] = np.arccos(cos_phi / (-2 * self.r * self.a * (1 - Tn[maxOrLess])))  # noqa

        return ans


class PhotonicNeuron:
    """
    A simple, time-independent model of a neuron.
    """
    def __init__(self, phaseShifts, outputGain):
        self.phaseShifts = np.asarray(phaseShifts)
        self.outputGain = outputGain
        self.inputSize = phaseShifts.size

        mrr = MRRTransferFunction()
        self._throughput = mrr.throughput(self.phaseShifts)
        self._dropput = mrr.dropput(self.phaseShifts)

    def step(self, intensities):
        intensities = np.asarray(intensities)
        if intensities.size != self.inputSize:
            raise AssertionError(
                    "Number of inputs ({}) is not "
                    "equal to  number of weights ({})".format(
                        intensities.size, self.inputSize))

        summedThroughput = np.dot(intensities, self._throughput)
        summedDropput = np.dot(intensities, self._dropput)
        photodiodeVoltage = summedDropput - summedThroughput

        return self.outputGain * photodiodeVoltage


class PhotonicNeuronArray:
    def __init__(self, inputShape, connections, neurons, sharedCounts):
        self.inputShape = inputShape
        assert neurons.shape == connections.shape
        self.connections = connections
        assert inputShape == sharedCounts.shape
        self.sharedCounts = sharedCounts
        self.neurons = neurons
        self._output = np.empty(self.connections.shape)

    def step(self, intenstiyMatrix):
        intenstiyMatrix = np.asarray(intenstiyMatrix)
        if intenstiyMatrix.shape != self.inputShape:
            raise AssertionError(
                "Input shape {} is not "
                "equal to array shape {}").format(
                    intenstiyMatrix.shape,
                    self.inputShape
                )

        for row in range(self.connections.shape[0]):
            for col in range(self.connections.shape[1]):
                conn = self.connections[row, col]
                inputs = intenstiyMatrix[conn[:, 0], conn[:, 1]]
                sharedCount = self.sharedCounts[conn[:, 0], conn[:, 1]]
                self._output[row, col] = self.neurons[row, col].step(
                    (inputs / sharedCount).ravel())

        return self._output


class LaserDiodeArray:
    """
    An array of laser diodes.
    """
    def __init__(self, shape, outputShape, connections, power):
        self.shape = shape
        self.connections = connections
        self._output = np.ones(outputShape) * power

    def step(self):
        return self._output


class ModulatorArray:
    """
    An array of photonic modulators.
    """
    def __init__(self, phaseShifts):
        self.phaseShifts = np.asarray(phaseShifts)
        self.inputShape = phaseShifts.shape
        self.mrm = MRMTransferFunction()
        self._throughput = self.mrm.throughput(self.phaseShifts)

    def step(self, intensities):
        intensities = np.asarray(intensities)
        if intensities.shape != self.inputShape:
            raise AssertionError(
                    "Input shape {} is not "
                    "equal to modulator shape {}").format(
                        intensities.shape,
                        self.inputShape
                    )

        return self._throughput * intensities

    def updatePhaseShifts(self, newPhaseShifts):
        self.phaseShifts = np.asarray(newPhaseShifts)
        self._throughput = self.mrm.throughput(self.phaseShifts)
