"""
Concept: Neuron

A neuron receives multiple input values, applies a corresponding
weight to each input, adds a bias, and passes the result through
an activation function.

The raw output is calculated as:

    raw_output = sum(weight × input) + bias

For example:

    inputs  = [x1, x2]
    weights = [w1, w2]

    raw_output = (w1 × x1) + (w2 × x2) + bias

The activation function then transforms the raw output:

    output = activation(raw_output)

During backpropagation, the neuron calculates:

    weight gradients
    bias gradient
    input gradients

Each input has its own weight and corresponding weight gradient.
"""

from .activation import Activation


class Neuron:
    """A trainable neuron with multiple inputs, weights, bias, and activation."""

    def __init__(
        self,
        weights: list[float],
        bias: float = 0.0,
        activation: Activation | None = None,
    ):
        self.weights = weights
        self.bias = bias
        self.gradient = [0.0] * len(weights)
        self.bias_gradient = 0.0
        self._input = [0.0] * len(weights)
        self.activation = activation or Activation()

    def forward(self, input_values: list[float]) -> float:
        """Calculate the neuron output."""
        self._input = input_values

        raw_output = sum(
            weight * input_value
            for weight, input_value in zip(
                self.weights,
                input_values,
                strict=True,
            )
        ) + self.bias

        return self.activation.forward(raw_output)

    def backward(self, output_gradient: float) -> list[float]:
        """
        Calculate gradients for weights, bias, and inputs.

        Returns:
            The gradients with respect to the inputs.
        """
        activation_gradient = self.activation.backward(
            output_gradient
        )

        self.gradient = [
            activation_gradient * input_value
            for input_value in self._input
        ]

        self.bias_gradient = activation_gradient

        input_gradients = [
            activation_gradient * weight
            for weight in self.weights
        ]

        return input_gradients