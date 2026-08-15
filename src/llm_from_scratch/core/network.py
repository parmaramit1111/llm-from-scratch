"""
Concept: Network

A network is a sequence of layers that process data one layer
at a time.

The output of one layer becomes the input to the next layer.

Forward propagation:

    Input
      ↓
    Layer 1
      ↓
    Layer 2
      ↓
    Output

Backward propagation works in the reverse order:

    Output gradients
          ↓
       Layer 2
          ↓
       Layer 1
          ↓
     Input gradients

The network coordinates the flow of data and gradients between
layers but does not perform the mathematical calculations itself.
"""

from .layer import Layer

class Network:
    """A sequence of layers that process data sequentially."""

    def __init__(self, layers: list[Layer]):
        self.layers = layers

    def forward(self, input_values: list[float]) -> list[float]:
        """Pass inputs sequentially through every layer."""
        activations = input_values
        for layer in self.layers:
            activations = layer.forward(activations)
        return activations

    def backward(self, output_gradients: list[float]) -> list[float]:
        """Pass gradients sequentially through every layer in reverse."""
        gradients = output_gradients
        for layer in reversed(self.layers):
            gradients = layer.backward(gradients)
        return gradients

