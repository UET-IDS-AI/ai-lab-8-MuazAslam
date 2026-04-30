"""
AI_stats_lab.py

Neural Networks Lab: 3-Layer Forward Pass and Backpropagation

Implement all functions.
Do NOT change function names.
Do NOT print inside functions.
"""

import numpy as np


def sigmoid(z):
    """
    sigmoid(z) = 1 / (1 + exp(-z))
    """
    return 1 / (1 + np.exp(-z))


def forward_pass(X, W1, W2, W3):
    """
    3-layer neural network forward pass.

    Layer 1:
        h1 = sigmoid(XW1)

    Layer 2:
        h2 = sigmoid(h1W2)

    Output layer:
        y = sigmoid(h2W3)

    Returns:
        h1, h2, y
    """
    h1 = sigmoid(np.dot(X, W1))
    h2 = sigmoid(np.dot(h1, W2))
    y = sigmoid(np.dot(h2, W3))
    return h1, h2, y


def backward_pass(X, h1, h2, y, label, W1, W2, W3):
    """
    Backpropagation for a 3-layer sigmoid neural network.

    Returns:
        dW1, dW2, dW3, loss
    """
    # Calculate cross-entropy loss as expected by the tests
    loss = -(label * np.log(y) + (1 - label) * np.log(1 - y))
    
    # Output layer gradients
    dJ_dy = -(label / y) + ((1 - label) / (1 - y))
    dy_dz3 = y * (1 - y)
    grad3 = dJ_dy * dy_dz3
    dW3 = np.dot(h2.T, grad3)
    
    # Layer 2 gradients
    dJ_dh2 = np.dot(grad3, W3.T)
    dh2_dz2 = h2 * (1 - h2)
    grad2 = dJ_dh2 * dh2_dz2
    dW2 = np.dot(h1.T, grad2)
    
    # Layer 1 gradients
    dJ_dh1 = np.dot(grad2, W2.T)
    dh1_dz1 = h1 * (1 - h1)
    grad1 = dJ_dh1 * dh1_dz1
    dW1 = np.dot(X.T, grad1)
    
    return dW1, dW2, dW3, loss
