"""
Concept: Neuron

A neuron is a small mathematical unit that takes an input,
applies a weight, and produces an output.

General form:

    output = (weight × input) + bias

For our first experiment, we intentionally keep it simple
and do not use bias yet:

    output = weight × input

The weight is the parameter that our model will eventually
learn during training.
"""


class Neuron:
    """A simple trainable neuron."""

    def __init__(self, weight: float):
        self.weight = weight
        self.gradient = 0.0
        self._input = 0.0

    def forward(self, input_value: float) -> float:
        """Calculate the output for a given input."""
        self._input = input_value
        return self.weight * input_value

    def backward(self, output_gradient: float) -> float:
        """
        Calculate the gradient for the neuron's weight.

        The output gradient represents how the loss changes
        with respect to the neuron's output.
        """
        self.gradient = output_gradient * self._input
        return self.gradient