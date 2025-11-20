
import numpy as np
from py_satl.uniform import uniformity_test

def test_uniform_ks_rejects_nonuniform():
    np.random.seed(0)
    normal_sample = np.random.normal(loc=0.5, scale=0.15, size=500)

    normal_sample = np.clip(normal_sample, 0, 1)
    stat, p = uniformity_test(normal_sample, method='ks')
    assert p < 0.05  # should usually reject uniform

def test_uniform_ks_accepts_uniform():
    np.random.seed(1)
    uni = np.random.uniform(0, 1, 500)
    stat, p = uniformity_test(uni, method='ks')
    assert p > 0.01

def test_uniform_chi2():
    np.random.seed(2)
    uni = np.random.uniform(0, 1, 500)
    stat, p = uniformity_test(uni, method='chi2', bins=10)
    assert p > 0.01
