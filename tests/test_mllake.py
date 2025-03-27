import pytest
import numpy as np
from mllake.classification import Classifier


def test_classification():
    X = np.random.rand(100, 5)
    y = np.random.randint(2, size=100)
    clf = Classifier()
    clf.fit(X, y)
    predictions = clf.predict(X)
    assert len(predictions) == 100

