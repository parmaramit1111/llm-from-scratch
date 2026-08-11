"""
Tests for the Neuron class.

The first test verifies the basic forward calculation:

    output = weight × input
"""

from llm_from_scratch.core.neuron import Neuron


def test_neuron_backward():
    """A neuron should calculate its weight gradient correctly."""
    neuron = Neuron(weight=2.0, bias=0.0)

    neuron.forward(3.0)

    gradient = neuron.backward(
        output_gradient=-6.0,
    )

    assert gradient == -18.0
    assert neuron.gradient == -18.0

def test_neuron_with_bias():
    """A neuron should include bias in its output."""
    neuron = Neuron(weight=3.0, bias=2.0)

    result = neuron.forward(4.0)

    assert result == 14.0


def test_neuron_backward_with_bias():
    """A neuron should calculate weight and bias gradients."""
    neuron = Neuron(weight=3.0, bias=2.0)

    neuron.forward(4.0)

    weight_gradient = neuron.backward(
        output_gradient=-2.0,
    )

    assert weight_gradient == -8.0
    assert neuron.gradient == -8.0
    assert neuron.bias_gradient == -2.0