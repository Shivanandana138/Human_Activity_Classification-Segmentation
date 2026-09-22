import numpy as np

from One_HMM.src.hmm.one_hmm import (
    create_hmm,
    train_one_hmm,
)


def test_create_hmm():

    model = create_hmm()

    assert model.n_components == 6


def test_train_one_hmm():

    X = np.random.RandomState(42).randn(
        100,
        561
    )

    model = train_one_hmm(X)

    assert model is not None
    assert model.n_components == 6