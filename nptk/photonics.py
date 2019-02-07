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
        cos_phi = ((1 - self.r1**2) * (1 - self.r2**2) * self.a / Td - 1 - (self.r1 * self.r2 * self.a)**2) # noqa
        return np.arccos(cos_phi / (-2 * self.r1 * self.r2 * self.a))


class MRMTransferFunction:
    """
    Computes the transfer function of a microring modulator (MRM).
    """
    def __init__(self, a=0.9, r=0.9):
        self.a = a
        self.r = r

    def throughput(self, phi):
        I_pass = self.a**2 - 2 * self.r * self.a * np.cos(phi) + self.r**2
        I_input = 1 - 2 * self.a * self.r * np.cos(phi) + (self.r * self.a)**2
        return I_pass / I_input

    def phaseFromThroughput(self, Tn):
        cos_phi = Tn * (1 + (self.r * self.a)**2) - self.a**2 - self.r**2
        return np.arccos(cos_phi / (-2 * self.r * self.a * (1 - Tn)))


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
    def __init__(self, inputShape, connections, neurons):
        self.inputShape = inputShape
        assert neurons.shape == connections.shape
        self.connections = connections
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
                connection = self.connections[row, col]
                inputs = intenstiyMatrix[connection[:, 0], connection[:, 1]]
                self._output[row, col] = self.neurons[row, col].step(
                    inputs.ravel())

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
