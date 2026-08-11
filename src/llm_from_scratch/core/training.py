"""
Concept: Training

Training repeatedly runs the learning cycle:

    Forward → Loss → Backward → Update

Each iteration gives the model an opportunity
to reduce its prediction error.
"""

from llm_from_scratch.core.loss import MeanSquaredError
from llm_from_scratch.core.neuron import Neuron
from llm_from_scratch.core.optimizer import GradientDescent


class Trainer:
    """Train a neuron using loss, gradients, and an optimizer."""

    def __init__(
        self,
        neuron: Neuron,
        loss_function: MeanSquaredError,
        optimizer: GradientDescent,
    ):
        self.neuron = neuron
        self.loss_function = loss_function
        self.optimizer = optimizer

    def train_step(
        self,
        input_value: float,
        target: float,
    ) -> float:
        """
        Perform one complete training step.

        Returns:
            The loss calculated for this step.
        """

        # 1. Forward pass
        prediction = self.neuron.forward(input_value)

        # 2. Calculate loss
        loss = self.loss_function.forward(
            prediction=prediction,
            target=target,
        )

        # 3. Calculate gradient of the loss
        loss_gradient = self.loss_function.backward(
            prediction=prediction,
            target=target,
        )

        # 4. Propagate gradient back to the neuron
        self.neuron.backward(loss_gradient)

        # 5. Update the neuron's weight
        self.neuron.weight = self.optimizer.step(
            parameter=self.neuron.weight,
            gradient=self.neuron.gradient,
        )

        return loss