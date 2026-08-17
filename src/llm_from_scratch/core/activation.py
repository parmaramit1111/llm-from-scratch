"""
Concept: Activation Function

An activation function transforms the raw output of a neuron
before passing it to the next layer.

Without an activation function, a neural network remains
effectively linear, even when multiple linear layers are used.

For our first activation function, we use ReLU
(Rectified Linear Unit):

    ReLU(x) = max(0, x)

ReLU passes positive values unchanged and converts
negative values to zero.

During backpropagation:

    x > 0  → gradient passes through
    x <= 0 → gradient becomes zero
"""

class Activation:
    """
    Apply a configured activation function and calculate its gradient.

    Supported activations:

        ReLU:
            forward(x)  = max(0, x)
            backward    = 1 when x > 0, otherwise 0

        Linear:
            forward(x)  = x
            backward    = 1
    """

    def __init__(self, name: str = "relu"):
        self.name = name
        self.raw_input = 0.0

    def forward(self, raw_input: float) -> float:
        """Apply the configured activation to the raw neuron output."""
        self.raw_input = raw_input

        if self.name == "linear":
            return raw_input

        if raw_input > 0:
            return raw_input

        return 0.0

    def backward(self, gradient: float) -> float:
        """Propagate the gradient through the configured activation."""
        if self.name == "linear":
            return gradient

        if self.raw_input > 0:
            return gradient

        return 0.0