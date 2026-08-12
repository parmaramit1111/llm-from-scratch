"""
Concept: Neuron

A neuron takes an input, applies a weight, and adds a bias.

    output = (weight × input) + bias

Weight controls how strongly the input affects the output.

Bias shifts the output up or down independently of the input.

Both weight and bias are trainable parameters.
"""


class Neuron:
    """A simple trainable neuron with weight and bias."""

    def __init__(self, weight: float, bias: float = 0.0):
        self.weight = weight
        self.bias = bias
        self.gradient = 0.0
        self.bias_gradient = 0.0
        self._input = 0.0

    def forward(self, input_value: float) -> float:
        """Calculate the neuron output."""
        self._input = input_value
        return self.weight * input_value + self.bias

    def backward(self, output_gradient: float) -> float:
        """
        Calculate gradients for weight, bias, and input.

        Returns:
            The gradient with respect to the input.
        """
        self.gradient = output_gradient * self._input
        self.bias_gradient = output_gradient

        input_gradient = output_gradient * self.weight

        return input_gradient