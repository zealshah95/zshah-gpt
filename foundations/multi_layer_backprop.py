import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self, 
                              x: List[float], 
                              W1: List[List[float]], b1: List[float], 
                              W2: List[List[float]], b2: List[float], 
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)

        # Convert all inputs to NumPy arrays
        x = np.array(x, dtype=float) # (input_size,)
        W1 = np.array(W1, dtype=float) # (hidden_size, input_size)
        b1 = np.array(b1, dtype=float) # (hidden_size,)
        W2 = np.array(W2, dtype=float) # (output_size, hidden_size)
        b2 = np.array(b2, dtype=float) # (output_size,)
        y_true = np.array(y_true, dtype=float) # (output_size,)

        # reshape 1-D arrays to 2-D for making computation more intuitive
        x = x.reshape(-1,1) if x.ndim==1 else x
        b1 = b1.reshape(-1,1) if b1.ndim==1 else b1
        b2 = b2.reshape(-1,1) if b2.ndim==1 else b2
        y_true = y_true.reshape(-1,1) if y_true.ndim==1 else y_true

        # forward pass
        z1 = W1 @ x + b1
        a1 = np.maximum(z1, 0)
        z2 = W2 @ a1 + b2

        L = np.mean((z2 - y_true) ** 2)

        # backward pass
        n = len(y_true) if y_true.ndim > 0 else 1
        dL_dz2 = (z2 - y_true) * 2 / n
        
        dL_dW2 = dL_dz2 @ a1.T
        dL_db2 = dL_dz2
        dL_da1 = W2.T @ dL_dz2

        dL_dz1 = dL_da1 * np.where(z1 > 0, 1, 0)
        
        dL_dW1 = dL_dz1 @ x.T
        dL_db1 = dL_dz1

        # reshape to original shape before returning
        return {
            "loss": round(float(L), 4),
            "dW1": np.round(dL_dW1, 4).tolist(),
            "db1": np.round(dL_db1.reshape(-1), 4).tolist(),
            "dW2": np.round(dL_dW2, 4).tolist(),
            "db2": np.round(dL_db2.reshape(-1), 4).tolist()
        }
