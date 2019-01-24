import numpy as np
from nptk.mappers import NeuronMapper


def test_mappers():
    weights = np.array([1, 1])
    print(weights.size)
    two_sum = NeuronMapper.build(weights)
    print(two_sum)

    assert False
