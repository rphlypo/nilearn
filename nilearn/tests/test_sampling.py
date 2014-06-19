from sklearn.utils import check_random_state
import numpy as np
from ..sampling import _sample_exact, StratifiedLeaveOneLabelOut
from nose.tools import assert_true, assert_false, assert_raises


def test_StratifiedLeaveOneLabelOut(random_state=1):
    rg = check_random_state(random_state)
    y = rg.randint(5, size=(500,))
    y = np.concatenate([y, rg.randint(3, size=(500,))])
    bincount_y = _normalised_histogram(y)[1]
    labels = rg.randint(3, size=(1000,))
    stratlolo = StratifiedLeaveOneLabelOut(y, labels)
    for train_ix, test_ix in stratlolo:
        bincount_train = _normalised_histogram(y[train_ix])[1]
        KL_train = _KL_divergence(bincount_train, bincount_y)
        bincount_test = _normalised_histogram(y[test_ix])[1]
        KL_test = _KL_divergence(bincount_test, bincount_y)
        # for large amount of samples, this should be approx. exact
        assert_true(KL_test <= 1e-3)
        assert_true(KL_train <= 1e-3)


def test__sample_exact(random_state=1):
    rg = check_random_state(random_state)
    while True:
        try:
            y = rg.randint(4, size=(10,))
            unique_y, bincount_y = _normalised_histogram(y)
            z = rg.randint(4, size=(50,))
            bincount_z = _normalised_histogram(z)[1]
            if bincount_y.size == 4 and bincount_z.size == 4:
                raise StopIteration
        except StopIteration:
            break

    target = bincount_y / np.sum(bincount_y).astype(np.float)
    KL_before = _KL_divergence(bincount_z, target)

    z_ = z[_sample_exact(z, target, unique_y, rg)]
    bincount_z_ = _normalised_histogram(z_)[1]
    KL_after = _KL_divergence(bincount_z_, target)

    assert_true(KL_before - KL_after > 0.)

    # what if not all labels are present in a leave-one-label-out?
    z_no3 = z[z < 3]
    assert_raises(ValueError, _sample_exact, z_no3, target, unique_y, rg)


def _normalised_histogram(x):
    ''' helper function for test, returns estimate of p.m.f. '''
    unique_x, inverse_x = np.unique(x, return_inverse=True)
    bincount_x = np.bincount(inverse_x)
    return unique_x, bincount_x / np.sum(bincount_x.astype(np.float))


def _KL_divergence(p_data, p_model):
    '''
    returns KL divergence between two p.m.f.'s from p_data to p_model

    if p_data is empirical, and p_model theoretical, it is the log-likelihood
    '''
    return np.sum(p_data * np.log(p_data / p_model.astype(np.float)))


def _disp_bin_count(x):
    ''' helper function for debugging the test, displays histogram '''
    unique_x, bincount_x = _normalised_histogram(x)
    for x_, count in zip(unique_x, bincount_x):
        print '{}:\t{}\n'.format(x_, '|' * np.round(count * x.size))
