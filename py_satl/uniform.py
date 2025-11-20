
import numpy as np
from scipy import stats

def uniformity_test(data, method='ks', bins=10):

    data = np.asarray(data)

    if data.size == 0:
        raise ValueError("data must not be empty")
    if np.any(data < 0) or np.any(data > 1):
        raise ValueError("data must be in [0, 1] for these tests")

    if method == 'ks':

        stat, pval = stats.kstest(data, 'uniform')
        return stat, pval

    elif method == 'chi2':
        counts, edges = np.histogram(data, bins=bins, range=(0.0, 1.0))
        expected = np.ones(bins) * (data.size / bins)

        stat, pval = stats.chisquare(counts, f_exp=expected)
        return stat, pval

    else:
        raise ValueError("method must be 'ks' or 'chi2'")
