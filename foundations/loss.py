import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: clip y_pred to [1e-7, 1 - 1e-7] to avoid log(0)
        # return round(your_answer, 4)
        y_pred = y_pred.clip(1e-7, 1 - 1e-7)
        loss = -1.0 * np.mean(y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred))
        return round(loss, 4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: clip y_pred to [1e-7, 1 - 1e-7] to avoid log(0)
        # return round(your_answer, 4)
        
        y_pred = y_pred.clip(1e-7, 1 - 1e-7)
        # solution 1: [not great]
        # for each sample, obtain y_pred where y_true is 1 i.e.
        # for each sample, obtain prediction probability of the correct class
        # loss = -1.0 * np.mean(np.log(y_pred[np.where(y_true == 1)]))

        # solution 2: [straightforward]
        loss = -1.0 * np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        
        return round(loss, 4)
