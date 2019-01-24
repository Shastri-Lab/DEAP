import numpy as np
import matplotlib.pyplot as plt
from helpers import bisect_min


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
    A simple, time-independent photonic neuron.
    """
    def __init__(self, phaseShifts, outputGain):
        self.phi = np.asarray(phaseShifts)
        self.outputGain = outputGain

        self.mrr = MRRTransferFunction()
        self.inputSize = len(phaseShifts)

    def compute(self, intensities):
        intensities = np.asarray(intensities)
        assert intensities.size == self.inputSize

        summedThroughput = np.sum(intensities * self.mrr.throughput(self.phi))
        summedDropput = np.sum(intensities * self.mrr.dropput(self.phi))
        photodiodeVoltage = summedDropput - summedThroughput

        return self.outputGain * photodiodeVoltage


class NeuralMapper:
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
        phi = np.linspace(np.deg2rad(-181), 0, 1000000)
        Td = MRRTransferFunction().dropput(phi)
        desiredDropput = np.linspace(0, 1, precision)

        dropput = np.zeros(precision)
        phase = np.zeros(precision)

        for i in range(precision):
            index = bisect_min(Td, desiredDropput[i])
            phase[i], dropput[i] = phi[index], Td[index]
            print(desiredDropput[i], dropput[i])

        return phase, dropput

    _precision = 1001

    # Precomputed phase and dropput values for a given precision.
    # These will be populated when a neuron is created.
    _phase = None
    _dropput = None

    def setPrecision(newPrecision):
        assert 1 <= newPrecision
        if newPrecision % 2 == 0:
            newPrecision -= 1

        NeuralMapper._precision = newPrecision
        NeuralMapper._phase = None
        NeuralMapper._dropput = None

    def build(weights):
        if NeuralMapper._phase is None or NeuralMapper._dropput is None:
            # Compute phase and dropput mapping if it hasn't been done
            # already.
            NeuralMapper._phase, NeuralMapper._dropput = \
                NeuralMapper._precomputeDropputToPhaseMapping(
                        NeuralMapper._precision)

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
            index = bisect_min(NeuralMapper._dropput, desiredDropput)
            phaseShifts[i] = NeuralMapper._phase[index]

        return PhotonicNeuron(phaseShifts, outputGain)
