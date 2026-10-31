import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        #
        # Forward: z = dot(x, w) + b, y_hat = sigmoid(z)
        # Loss: L = 0.5 * (y_hat - y_true)^2
        # Return: (dL_dw rounded to 5 decimals, dL_db rounded to 5 decimals)
        
        # forward pass
        hpreact = np.matmul(x,w) + b
        out = 1/(1 + np.exp(-hpreact))
        L = 0.5 * ((out - y_true) ** 2)
        
        # backward pass
        dL_dout = out - y_true

        dout_dhpreact = out * (1-out)
        dL_dhpreact = dL_dout * dout_dhpreact
        
        dL_dw = dL_dhpreact * x
        dL_db = dL_dhpreact

        return (np.round(dL_dw, 5), np.round(dL_db, 5))
