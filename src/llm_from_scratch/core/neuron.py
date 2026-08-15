"""
Concept: Neuron

A neuron takes an input, applies a weight, and adds a bias.

    output = (weight × input) + bias

Weight controls how strongly the input affects the output.

Bias shifts the output up or down independently of the input.

Both weight and bias are trainable parameters.
"""

from .activation import Activation


class Neuron:
    """A simple trainable neuron with weight and bias."""

    def __init__(
        self,
        weight: float,
        bias: float = 0.0,
        activation: Activation | None = None,
    ):
        self.weight = weight
        self.bias = bias
        self.gradient = 0.0
        self.bias_gradient = 0.0
        self._input = 0.0
        self.activation = activation or Activation()


    def forward(self, input_value: float) -> float:
        """Calculate the neuron output."""
        self._input = input_value

        raw_output = self.weight * input_value + self.bias

        return self.activation.forward(raw_output)

    def backward(self, output_gradient: float) -> float:
        """
        Calculate gradients for weight, bias, and input.

        Returns:
            The gradient with respect to the input.
        """
        activation_gradient = self.activation.backward(
            output_gradient
        )
        self.gradient = activation_gradient * self._input
        self.bias_gradient = activation_gradient

        input_gradient = activation_gradient * self.weight

        return input_gradient