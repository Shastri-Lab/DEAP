import numpy as np
from nptk.mappers import NeuronMapper


def test_NeuronMapperTwoSum():
    weights = np.array([1, 1])
    neuron = NeuronMapper.build(weights)
    assert neuron.outputGain == 1

    for v in ([1, 1], [2, 2], [-1, 1], [10, 1]):
        computed = neuron.compute(v)
        expected = np.dot(weights, v)
        assert np.abs(computed - expected < 1e-4)


def test_NeuronMapperUnity():
    weights = np.array([1])
    unityNeuron = NeuronMapper.build(weights)
    assert unityNeuron.outputGain == 1

    for v in ([1], [2], [-1], [10]):
        computed = unityNeuron.compute(v)
        expected = np.dot(weights, v)
        assert np.abs(computed - expected < 1e-4)


def test_NeuronMapperNegative():
    weights = np.array([-1])
    neuron = NeuronMapper.build(weights)
    assert neuron.outputGain == 1

    for v in ([1], [2], [-1], [10]):
        computed = neuron.compute(v)
        expected = np.dot(weights, v)
        err = (expected - computed) / expected
        assert np.abs(err < 1e-3)


def test_NeuronMapperNull():
    weights = np.array([0])
    unityNeuron = NeuronMapper.build(weights)
    assert unityNeuron.outputGain == 1

    for v in ([1], [2], [-1], [10]):
        computed = unityNeuron.compute(v)
        expected = np.dot(weights, v)
        assert np.abs(computed - expected < 2e-3)


def test_NeuronMapplerMultiple():
    weights = np.array([2, -3, 4, 0, 9])
    neuron = NeuronMapper.build(weights)

    for v in ([1, 10, 3, 2, 0], [1, 1, 1, 1, 1], [-7, 0.214, 22, 0.7, 2]):
        computed = neuron.compute(v)
        expected = np.dot(weights, v)
        err = (expected - computed) / expected
        assert err < 2e-3
