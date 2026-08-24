"""
Tests for the Neuron class.

The tests verify:

    output = activation(sum(weight × input) + bias)

The tests cover:

- Single and multiple inputs
- Bias
- Weight, bias, and input gradients
- ReLU forward and backward behavior
- Mismatched input and weight validation
"""

from llm_from_scratch.core.neuron import Neuron


def test_neuron_backward():
    """A neuron should calculate weight, bias, and input gradients."""
    neuron = Neuron(weights=[2.0], bias=0.0)

    neuron.forward([3.0])

    input_gradient = neuron.backward(
        output_gradient=-6.0,
    )

    # dLoss/dWeight = activation_gradient × input
    assert neuron.gradient == [-18.0]

    # dLoss/dBias = activation_gradient
    assert neuron.bias_gradient == -6.0

    # dLoss/dInput = activation_gradient × weight
    assert input_gradient == [-12.0]

def test_neuron_with_bias():
    """A neuron should include bias in its output."""
    neuron = Neuron(weights=[3.0], bias=2.0)

    result = neuron.forward([4.0])

    assert result == 14.0


def test_neuron_backward_with_bias():
    """A neuron should calculate weight, bias, and input gradients."""
    neuron = Neuron(weights=[3.0], bias=2.0)

    neuron.forward([4.0])

    input_gradient = neuron.backward(
        output_gradient=-2.0,
    )

    assert neuron.gradient == [-8.0]
    assert neuron.bias_gradient == -2.0
    assert input_gradient == [-6.0]


def test_neuron_positive_backward():
    """A neuron should pass gradients when its raw output is positive."""
    neuron = Neuron(weights=[2.0], bias=0.0)

    neuron.forward([3.0])

    input_gradient = neuron.backward(
        output_gradient=-6.0,
    )

    assert neuron.gradient == [-18.0]
    assert neuron.bias_gradient == -6.0
    assert input_gradient == [-12.0]

def test_neuron_negative_backward():
    """A neuron should block gradients when its raw output is negative."""
    neuron = Neuron(weights=[2.0], bias=0.0)

    neuron.forward([-3.0])

    input_gradient = neuron.backward(
        output_gradient=-6.0,
    )

    assert neuron.gradient == [0.0]
    assert neuron.bias_gradient == 0.0
    assert input_gradient == [0.0]

def test_neuron_zero_backward():
    """A neuron should block gradients when its raw output is zero."""
    neuron = Neuron(weights=[2.0], bias=0.0)

    neuron.forward([0.0])

    input_gradient = neuron.backward(
        output_gradient=3.0,
    )

    assert neuron.gradient == [0.0]
    assert neuron.bias_gradient == 0.0
    assert input_gradient == [0.0]


def test_neuron_with_multiple_inputs():
    """A neuron should calculate a weighted sum from multiple inputs."""
    neuron = Neuron(
        weights=[2.0, 3.0],
        bias=1.0,
    )

    result = neuron.forward([4.0, 5.0])

    assert result == 24.0

def test_neuron_backward_with_multiple_inputs():
    """A neuron should calculate a gradient for each weight."""
    neuron = Neuron(
        weights=[2.0, 3.0],
        bias=1.0,
    )

    neuron.forward([4.0, 5.0])

    input_gradients = neuron.backward(
        output_gradient=2.0,
    )

    assert neuron.gradient == [8.0, 10.0]
    assert neuron.bias_gradient == 2.0
    assert input_gradients == [4.0, 6.0]

def test_neuron_rejects_mismatched_inputs_and_weights():
    """A neuron should reject mismatched input and weight lengths."""
    neuron = Neuron(
        weights=[2.0, 3.0],
        bias=1.0,
    )

    try:
        neuron.forward([4.0])
        assert False
    except ValueError:
        pass
