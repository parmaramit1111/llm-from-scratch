"""
Tests for the Neuron class.

The first test verifies the basic forward calculation:

    output = weight × input
"""

from llm_from_scratch.core.neuron import Neuron


def test_neuron_backward():
    """A neuron should calculate weight, bias, and input gradients."""
    neuron = Neuron(weight=2.0, bias=0.0)

    neuron.forward(3.0)

    input_gradient = neuron.backward(
        output_gradient=-6.0,
    )

    # dLoss/dWeight = output_gradient × input
    assert neuron.gradient == -18.0

    # dLoss/dBias = output_gradient
    assert neuron.bias_gradient == -6.0

    # dLoss/dInput = output_gradient × weight
    assert input_gradient == -12.0

def test_neuron_with_bias():
    """A neuron should include bias in its output."""
    neuron = Neuron(weight=3.0, bias=2.0)

    result = neuron.forward(4.0)

    assert result == 14.0


def test_neuron_backward_with_bias():
    """A neuron should calculate weight, bias, and input gradients."""
    neuron = Neuron(weight=3.0, bias=2.0)

    neuron.forward(4.0)

    input_gradient = neuron.backward(
        output_gradient=-2.0,
    )

    assert neuron.gradient == -8.0
    assert neuron.bias_gradient == -2.0
    assert input_gradient == -6.0
