import numpy as np


class SupervisedData:
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self._X = X
        self._y = y

    @property
    def X(self):
        return self._X

    @property
    def y(self):
        return self._y
