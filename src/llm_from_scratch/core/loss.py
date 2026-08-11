"""
Concept: Loss

A model produces a prediction, but we need a way to measure
how far that prediction is from the expected target.

For our first experiment, we use Mean Squared Error (MSE):

    loss = (prediction - target)²

A smaller loss means the prediction is closer to the target.
"""


class MeanSquaredError:
    """Calculate prediction error using Mean Squared Error."""

    def forward(self, prediction: float, target: float) -> float:
        """Calculate the squared error between prediction and target."""
        error = prediction - target
        return error**2

    def backward(self, prediction: float, target: float) -> float:
        """Calculate how the loss changes with the prediction."""
        return 2 * (prediction - target)