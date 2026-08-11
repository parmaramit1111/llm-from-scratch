"""
Tests for the Neuron class.

The first test verifies the basic forward calculation:

    output = weight × input
"""

from llm_from_scratch.core.neuron import Neuron


def test_neuron_backward():
    """A neuron should calculate its weight gradient correctly."""
    neuron = Neuron(weight=2.0)

    neuron.forward(3.0)

    gradient = neuron.backward(
        output_gradient=-6.0,
    )

    assert gradient == -18.0
    assert neuron.gradient == -18.0