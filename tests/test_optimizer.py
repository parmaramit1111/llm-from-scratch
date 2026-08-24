"""
Tests for the GradientDescent optimizer.

The optimizer should update a parameter using:

    new_parameter = old_parameter - learning_rate × gradient
"""

from llm_from_scratch.core.optimizer import GradientDescent


def test_gradient_descent_step():
    """Gradient descent should update the parameter correctly."""
    optimizer = GradientDescent(learning_rate=0.1)

    result = optimizer.step(
        parameter=1.0,
        gradient=-2.0,
    )

    assert result == 1.2


def test_gradient_descent_moves_parameter_down():
    """A positive gradient should decrease the parameter."""
    optimizer = GradientDescent(learning_rate=0.1)

    result = optimizer.step(
        parameter=2.0,
        gradient=3.0,
    )

    assert result == 1.7