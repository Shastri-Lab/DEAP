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


class PhotonicElement:
    def step(self, *args):
        raise NotImplementedError()


class PhotonicNeuron(PhotonicElement):
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


class LaserDiodeArray(PhotonicElement):
    """
    An array of laser diodes.
    """
    def __init__(self, shape, outputShape, connections, power):
        assert type(shape) is tuple
        assert type(outputShape) is tuple
        self.shape = shape
        self._output = np.ones(outputShape) * power
        self.connections = connections

    def step(self):
        return self._output


class ModulatorArray(PhotonicElement):
    """
    An array of photonic modulators.
    """
    def __init__(self, phaseShifts):
        self.phaseShifts = np.asarray(phaseShifts)

        self.inputShape = phaseShifts.shape
        mrm = MRMTransferFunction()
        self._throughput = mrm.throughput(self.phaseShifts)

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
