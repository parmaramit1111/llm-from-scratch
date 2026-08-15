"""
Experiment: Multiple Layers

Demonstrate forward and backward propagation through
a network containing multiple layers.

Architecture:

    Input
      ↓
    Layer 1
      ↓
    Layer 2
      ↓
    Output

Backward propagation travels through the layers
in the reverse order.
"""

from llm_from_scratch.core.layer import Layer
from llm_from_scratch.core.network import Network
from llm_from_scratch.core.neuron import Neuron


def main():
    """Run the multiple-layer experiment."""

    layer_1 = Layer([
        Neuron(weights=[0.5, 0.2], bias=1.0),
        Neuron(weights=[0.3, 0.8], bias=0.5),
        Neuron(weights=[0.7, 0.4], bias=0.0),
    ])

    layer_2 = Layer([
        Neuron(weights=[0.2, 0.4, 0.6], bias=1.0),
        Neuron(weights=[0.5, 0.3, 0.1], bias=0.5),
    ])

    network = Network([
        layer_1,
        layer_2,
    ])

    input_values = [2.0, 3.0]

    print("=== Multiple Layer Experiment ===")
    print()
    print(f"Input: {input_values}")

    # Forward propagation
    predictions = network.forward(input_values)

    print(f"Output: {predictions}")

    # Backward propagation
    output_gradients = [1.0, -2.0]

    input_gradients = network.backward(
        output_gradients=output_gradients,
    )

    print(f"Output gradients: {output_gradients}")
    print(f"Input gradients: {input_gradients}")


if __name__ == "__main__":
    main()