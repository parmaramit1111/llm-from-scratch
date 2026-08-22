"""
Tests for MeanSquaredError.

The test verifies:

    loss = (prediction - target)²
"""

import pytest

from llm_from_scratch.core.loss import MeanSquaredError, CrossEntropy


def test_mean_squared_error():
    """Mean Squared Error should calculate the squared error correctly."""
    loss_function = MeanSquaredError()

    result = loss_function.forward(
        predictions=[5.0],
        targets=[6.0],
    )

    assert result == 1.0


def test_mean_squared_error_for_correct_prediction():
    """Loss should be zero when prediction matches the target."""
    loss_function = MeanSquaredError()

    result = loss_function.forward(
        predictions=[6.0],
        targets=[6.0],
    )

    assert result == 0.0

def test_mean_squared_error_forward():
    """MSE should return the average squared error."""
    loss = MeanSquaredError()

    result = loss.forward(
        predictions=[9.0, 12.0, 4.0],
        targets=[10.0, 10.0, 3.0],
    )

    assert result == 2.0

def test_mean_squared_error_backward():
    """MSE should calculate a gradient for each prediction."""
    loss = MeanSquaredError()

    gradients = loss.backward(
        predictions=[9.0, 12.0, 4.0],
        targets=[10.0, 10.0, 3.0],
    )

    assert gradients == [
        -2.0 / 3.0,
        4.0 / 3.0,
        2.0 / 3.0,
    ]

def test_mean_squared_error_requires_matching_lengths():
    """MSE should reject predictions and targets of different lengths."""
    loss = MeanSquaredError()

    with pytest.raises(ValueError):
        loss.forward(
            predictions=[9.0, 12.0, 4.0],
            targets=[10.0, 10.0],
        )


def test_cross_entropy_backward():
    loss_function = CrossEntropy()

    gradient = loss_function.backward(
        probabilities=[0.05, 0.80, 0.10, 0.05],
        target_index=1,
    )

    assert gradient == [0.0, -1.25, 0.0, 0.0]


def test_cross_entropy_forward():
    loss_function = CrossEntropy()

    loss = loss_function.forward(
        probabilities=[0.05, 0.80, 0.10, 0.05],
        target_index=1,
    )

    assert abs(loss - 0.2231435513) < 1e-10