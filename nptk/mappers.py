import numpy as np
from nptk.photonics import MRRTransferFunction, PhotonicNeuron
from nptk.helpers import bisect_min


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
