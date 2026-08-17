"""
Unit Tests: Activation

These tests verify the forward and backward behavior of ReLU.

Forward:
    Positive values pass through unchanged.
    Negative and zero values become zero.

Backward:
    Positive inputs allow the gradient to pass through.
    Negative and zero inputs block the gradient.
"""

from llm_from_scratch.core.activation import Activation


def test_activation_positive():
    """ReLU should pass positive values unchanged."""
    activation = Activation()

    final_output = activation.forward(5.0)

    assert final_output == 5.0


def test_activation_negative():
    """ReLU should convert negative values to zero."""
    activation = Activation()

    final_output = activation.forward(-5.0)

    assert final_output == 0.0


def test_activation_zero():
    """ReLU should return zero when the input is zero."""
    activation = Activation()

    final_output = activation.forward(0.0)

    assert final_output == 0.0


def test_activation_positive_gradient():
    """ReLU should pass the gradient for positive input."""
    activation = Activation()

    activation.forward(5.0)
    final_gradient = activation.backward(3.0)

    assert final_gradient == 3.0


def test_activation_negative_gradient():
    """ReLU should block the gradient for negative input."""
    activation = Activation()

    activation.forward(-5.0)
    final_gradient = activation.backward(3.0)

    assert final_gradient == 0.0


def test_activation_zero_gradient():
    """ReLU should block the gradient when input is zero."""
    activation = Activation()

    activation.forward(0.0)
    final_gradient = activation.backward(3.0)

    assert final_gradient == 0.0

def test_linear_activation_forward():
    """A linear activation should return the raw input unchanged."""
    activation = Activation("linear")

    assert activation.forward(-3.0) == -3.0
    assert activation.forward(5.0) == 5.0

def test_linear_activation_backward():
    """A linear activation should pass gradients unchanged."""
    activation = Activation("linear")

    activation.forward(-3.0)

    assert activation.backward(7.0) == 7.0