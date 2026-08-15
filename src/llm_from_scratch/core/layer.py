"""
Concept: Layer

A layer is a collection of neurons that process the same input.

Each neuron has its own weight, bias, and activation function.

The layer coordinates the neurons but does not perform the
neuron's mathematical calculation or activation itself.

Example:

    input
      ↓
    ┌───┬───┬───┐
    ↓   ↓   ↓
   N1  N2  N3
    ↓   ↓   ↓
  Act  Act  Act
    ↓   ↓   ↓
   O1  O2  O3

Forward:
    Input → Neuron → Activation → Output

Backward:
    Output gradients
          ↓
    Neuron → Activation → Input gradients
"""


from llm_from_scratch.core.neuron import Neuron


class Layer:
    """A simple layer containing multiple neurons."""

    def __init__(self, neurons: list[Neuron]):
        self.neurons = neurons

    def forward(self, input_value: float) -> list[float]:
        """Pass the input through every neuron."""
        return [
            neuron.forward(input_value)
            for neuron in self.neurons
        ]

    def backward(self, output_gradients: list[float]) -> list[float]:
        """
        Pass each output gradient to its corresponding neuron.

        Returns:
            The gradients calculated by each neuron.
        """
        return [
            neuron.backward(output_gradient)
            for neuron, output_gradient in zip(
                self.neurons,
                output_gradients,
                strict=True,
            )
        ]