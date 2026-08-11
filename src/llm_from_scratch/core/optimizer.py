"""
Concept: Optimizer

An optimizer updates the parameters of a model using gradients.

For our first experiment, we use Gradient Descent:

    new_parameter = old_parameter - learning_rate × gradient

The gradient tells us which direction to move.
The learning rate controls how large the step should be.
"""


class GradientDescent:
    """Update model parameters using gradient descent."""

    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    def step(self, parameter: float, gradient: float) -> float:
        """Calculate the updated parameter."""
        return parameter - self.learning_rate * gradient