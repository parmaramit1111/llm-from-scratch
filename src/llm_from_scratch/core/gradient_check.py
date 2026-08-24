"""
Concept: Gradient Checking

Gradient checking verifies that an analytical gradient calculated
by backward propagation is correct.

It independently estimates the gradient using a numerical
finite-difference approximation and compares the two values.

Numerical gradient:

    gradient ≈
        [Loss(parameter + ε) - Loss(parameter - ε)]
        ---------------------------------------------
                       2 × ε

The gradient checker is a verification utility only.

It must not participate in normal model training.
"""


from collections.abc import Callable


class GradientChecker:
    """Verify analytical gradients using numerical finite differences."""

    def __init__(
        self,
        epsilon: float = 1e-6,
        tolerance: float = 1e-5,
    ):
        self.epsilon = epsilon
        self.tolerance = tolerance

    def numerical_gradient(
        self,
        parameter: float,
        loss_function: Callable[[float], float],
    ) -> float:
        """
        Calculate the numerical gradient for a parameter.

        Args:
            parameter:
                The parameter whose gradient is being checked.

            loss_function:
                A function that calculates the loss using the
                supplied parameter.

        Returns:
            The numerical approximation of the gradient.
        """

        loss_plus = loss_function(
            parameter + self.epsilon,
        )

        loss_minus = loss_function(
            parameter - self.epsilon,
        )

        return (loss_plus - loss_minus) / (2.0 * self.epsilon)

    def check(
        self,
        analytical_gradient: float,
        numerical_gradient: float,
    ) -> bool:
        """
        Compare analytical and numerical gradients.

        Returns:
            True when the gradients are within the configured tolerance.
        """

        return abs(analytical_gradient - numerical_gradient) < self.tolerance