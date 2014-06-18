from sklearn.utils import check_random_state
import numpy as np
from ..sampling import _sample_exact, StratifiedLeaveOneLabelOut
from nose.tools import assert_true, assert_false, assert_raises


def test_distribution():
    rg = check_random_state(1)
    y = rg.randint(5, size=(100,))
    labels = rg.randint(3, size=(100,))
    stratlolo = StratifiedLeaveOneLabelOut(y, y=labels)
    # for train_ix, test_ix in stratlolo:


def test__sample_exact(random_state=1):
    rg = check_random_state(random_state)
    while True:
        try:
            y = rg.randint(4, size=(10,))
            unique_y, inverse_y = np.unique(y, return_inverse=True)
            bincount_y = np.bincount(inverse_y)
            if bincount_y.size == 4:
                raise StopIteration
        except StopIteration:
            break

    target = bincount_y / np.sum(bincount_y).astype(np.float)
    z = rg.randint(4, size=(50,))
    bincount_z = _normalised_histogram(z)
    KL_before = _KL_divergence(bincount_z, target)

    z_ = z[_sample_exact(z, target, unique_y, rg)]
    bincount_z_ = _normalised_histogram(z_)
    KL_after = _KL_divergence(bincount_z_, target)

    assert_true(KL_before - KL_after > 0.)

    # what if not all labels are present in a leave-one-label-out?
    z_no3 = z[z < 3]
    assert_raises(ValueError, _sample_exact, z_no3, target, unique_y, rg)


def _normalised_histogram(x):
    x_bincount = np.bincount(np.unique(x, return_inverse=True)[1])
    return x_bincount / np.sum(x_bincount.astype(np.float))


def _KL_divergence(p_data, p_model):
    return np.sum(p_data * np.log(p_data / p_model.astype(np.float)))
