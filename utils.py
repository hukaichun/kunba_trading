from typing import Any
import numpy as np

class NaiveNormalizer:
    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        self._mean = mean
        self._std = std

    def __call__(self, X:np.ndarray):
        return (X-self._mean)/self._std
    

    @staticmethod
    def fit(all_X:np.ndarray):
        mean = all_X.mean(0)
        std = all_X.std(0)
        return NaiveNormalizer(mean, std)