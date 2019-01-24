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


class PhotonicNeuron:
    """
    A simple, time-independent model of a neuron.
    """
    def __init__(self, phaseShifts, outputGain):
        self.phi = np.asarray(phaseShifts)
        self.outputGain = outputGain

        self.inputSize = len(phaseShifts)
        mrr = MRRTransferFunction()
        self._throughput = mrr.throughput(self.phi)
        self._dropput = mrr.dropput(self.phi)

    def compute(self, intensities):
        intensities = np.asarray(intensities)
        assert intensities.size == self.inputSize

        summedThroughput = np.dot(intensities, self._throughput)
        summedDropput = np.dot(intensities, self._dropput)
        photodiodeVoltage = summedDropput - summedThroughput

        return self.outputGain * photodiodeVoltage
