"""
Concept: Layer

A layer is a collection of neurons that process the same input
vector.

Each neuron has its own weights, bias, and activation function.

Every neuron receives the same input vector but uses its own
weights to produce one output.

The layer coordinates the neurons but does not perform the
neuron's mathematical calculation or activation itself.

Example:

    Input
    [x1, x2]
       ↓
    ┌───────────────┐
    │     Layer     │
    │               │
    │ N1  N2  N3    │
    └───────────────┘
       ↓   ↓   ↓
      O1  O2  O3

Forward:

    Input Vector
         ↓
    Each Neuron
         ↓
    Output Vector

Backward:

    Output Gradients
         ↓
    Each Neuron
         ↓
    Input Gradients
         ↓
    Aggregate Gradients
         ↓
    Input Gradient Vector
"""


from llm_from_scratch.core.neuron import Neuron


class Layer:
    """A layer containing multiple neurons that share the same input vector."""

    def __init__(self, neurons: list[Neuron]):
        self.neurons = neurons

    def forward(self, input_values: list[float]) -> list[float]:
        """
        Pass each output gradient to its corresponding neuron.

        Returns:
            The gradients calculated by each neuron.
        """
        return [
            neuron.forward(input_values)
            for neuron in self.neurons
        ]

    def backward(self, output_gradients: list[float]) -> list[float]:
        """
        Pass each output gradient to its corresponding neuron
        and aggregate the gradients with respect to the inputs.

        Returns:
            The aggregated gradients with respect to the input vector.
        """

        gradients = [
            neuron.backward(output_gradient)
            for neuron, output_gradient in zip(
                self.neurons,
                output_gradients,
                strict=True,
            )
        ]

        return [
            sum(input_gradient)
            for input_gradient in zip(
                *gradients,
                strict=True,
            )
        ]
