import numpy as np
from nptk.photonics import MRRTransferFunction, PhotonicNeuron


def test_MRRTransferFunction():
    mrr = MRRTransferFunction(a=1, r1=0.9, r2=0.9)

    phi = np.linspace(-np.pi, np.pi, 1000)

    Tp = mrr.throughput(phi)
    Td = mrr.dropput(phi)

    summed = np.mean(Tp + Td)
    assert summed == 1


def test_Photonics():
    phase = np.deg2rad(np.array([180, 175, 10, 20, 0, -10, -15, 20]))
    inputs = [0, 1, 2, 3, 4, 5, 6, 7]
    gain = 1

    neuron = PhotonicNeuron(phase, gain)
    computed = neuron.compute(inputs)

    mrr = MRRTransferFunction()
    expected = gain * \
        (np.sum(inputs * mrr.dropput(phase)) -
            np.sum(inputs * mrr.throughput(phase)))

    assert computed == expected

    gain = 0
    neuron = PhotonicNeuron(phase, gain)
    computed = neuron.compute(inputs)
    assert computed == 0
