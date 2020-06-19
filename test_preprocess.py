import numpy as np

from preprocess import fix_data


def test_fix_data():
    n = 5
    x = list(np.random.random(n))

    assert np.allclose(x[2:-1], fix_data(x, 1, n-3))
    assert np.allclose(x[1:-1], fix_data(x, 1, n-2))
    assert np.allclose(x[1:], fix_data(x, 1, n-1))

    assert np.allclose(x, fix_data(x, 1, n))

    assert np.allclose(x+[0], fix_data(x, 1, n+1))
    assert np.allclose([0]+x+[0], fix_data(x, 1, n+2))
    assert np.allclose([0]+x+[0, 0], fix_data(x, 1, n+3))
