"""
Tests for MeanSquaredError.

The test verifies:

    loss = (prediction - target)²
"""

from llm_from_scratch.core.loss import MeanSquaredError


def test_mean_squared_error():
    """Mean Squared Error should calculate the squared error correctly."""
    loss_function = MeanSquaredError()

    result = loss_function.forward(
        prediction=5.0,
        target=6.0,
    )

    assert result == 1.0


def test_mean_squared_error_for_correct_prediction():
    """Loss should be zero when prediction matches the target."""
    loss_function = MeanSquaredError()

    result = loss_function.forward(
        prediction=6.0,
        target=6.0,
    )

    assert result == 0.0

def test_mean_squared_error_backward():
    """MSE backward should calculate the loss gradient correctly."""
    loss_function = MeanSquaredError()

    gradient = loss_function.backward(
        prediction=6.0,
        target=9.0,
    )

    assert gradient == -6.0