import numpy as np
from sklearn.utils import check_random_state
from sklearn.cross_validation import LeaveOneLabelOut


class StratifiedLeaveOneLabelOut():
    """ A simple object to perform stratified sampling w.r.t. y for leave one
    label out with respect to labels

    Arguments
    ---------
    y       : list or ndarray
        the classification labels (targets) which are to be stratified

    labels  : list or ndarray
        confounding labels, i.e., labels used in the leave-one-label-out

    n_iter  : unsigned int
        multiple draws of the stratified shuffle-and-split are possible

    n_draws : unisgned int
        two strategies are possible:
        - exact sampling (n_draws == 1), or
        - random sampling (n_draws > 1)
        exact sampling maintains the proportion of the samples, although no
        prediction is possible on the number of train/test samples retained;
        random sampling optimises the sample proportions for a desired number
        of train/test samples

    n_train : unsigned float in (0, 1]
        proportion of train samples in the leave-one-label-out to retain

    n_test  : unsigned float in (0, 1]
        proportion of test samples in the leave-one-label-out to retain

    random_state : integer or None
        seed of the random generator


    Returns
    -------
    list of tuples (train_ix, test_ix) satisfying a stratification of the
    leave-one-label-out sampler
    """
    def __init__(self, y, labels, n_iter=1, n_draws=10, n_train=.25,
                 n_test=.25, random_state=None):
        self.y = y
        self.labels = labels
        self.n_iter = n_iter
        self.n_draws = n_draws

        rg = check_random_state(random_state)

        y = np.asarray(y)
        labels = np.asarray(labels)
        # construct target distribution
        n_samples = y.shape[0]
        unique_y, y_inversed = np.unique(y, return_inverse=True)
        y_counts = np.bincount(y_inversed)
        py = y_counts / np.float(n_samples)

        lolo = LeaveOneLabelOut(labels)

        for train_ix, test_ix in lolo:
            try:
                if n_draws == 1:
                    train_ix_ = train_ix[_sample_exact(y[train_ix],
                                                       py, unique_y,
                                                       random_state=rg)]
                    test_ix_ = test_ix[_sample_exact(y[test_ix], py,
                                                     unique_y,
                                                     random_state=rg)]

                yield train_ix_, test_ix_

            except ValueError as ve:
                print ve.message
        # In the KFold it is merely a question of folding each y
        # this is true since the fold is an independent and balanced
        # confounding variable
        # this is no longer so for a pre-specified label (may be unbalanced, or
        # worse, dependent)
        # Should we run a dependence test of y on the labels?

        # get a sampling scheme:
        # for each leave-one-label-out split
        #     1. perform n_iter random samples/re-orderings
        #     2.


def _sample_exact(z, target, target_vars, random_state=None):
    """ sampling based on random pruning of samples

    input
    -----
    z   : ndarray
        pool of samples

    target: ndarray
        sample histogram
    """
    target = target / np.sum(target).astype(np.float)
    rg = check_random_state(random_state)
    unique_z, z_inversed = np.unique(z, return_inverse=True)
    set_diff = set(target_vars) - set(unique_z)
    # check whether labels are present in z
    if set_diff:
        raise ValueError('Truncated set does not contain all ' +
                         'labels from the population. ' +
                         'Missing labels: {}'.format(list(set_diff)))
    z_counts = np.bincount(z_inversed)
    target_counts = np.min(np.floor((z_counts - 1 / 2.) / target))
    target_counts = np.ceil(target * target_counts).astype(np.int)
    ix_list = list()
    for ix, z_ in enumerate(unique_z):
        ix_z_ = np.where(z == z_)[0]
        ix_list.append(ix_z_[rg.permutation(
            ix_z_.size)[:target_counts[ix]]])
    return np.concatenate(ix_list, axis=0)
