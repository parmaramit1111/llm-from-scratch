"""
Concept: Loss

A model produces predictions, but we need a way to measure
how far those predictions are from the expected targets.

For our first experiment, we use Mean Squared Error (MSE):

    MSE = average((prediction - target)²)

A smaller loss means the predictions are closer to the targets.

For multiple outputs, MSE calculates the average squared error
across all predictions.
"""


class MeanSquaredError:
    """Calculate prediction error using Mean Squared Error."""

    def forward(
        self,
        predictions: list[float],
        targets: list[float],
    ) -> float:
        """Calculate the mean squared error."""

        squared_errors = [
            (prediction - target) ** 2
            for prediction, target in zip(
                predictions,
                targets,
                strict=True,
            )
        ]

        return sum(squared_errors) / len(squared_errors)

    def backward(
        self,
        predictions: list[float],
        targets: list[float],
    ) -> list[float]:
        """Calculate the gradient of the loss for each prediction."""

        count = len(predictions)

        return [
            2 * (prediction - target) / count
            for prediction, target in zip(
                predictions,
                targets,
                strict=True,
            )
        ]