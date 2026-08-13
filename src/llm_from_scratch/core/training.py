"""
Concept: Training

Training repeatedly runs the learning cycle:

    Forward → Loss → Backward → Update

Each iteration gives the model an opportunity
to reduce its prediction error.
"""

from llm_from_scratch.core.loss import MeanSquaredError
from llm_from_scratch.core.layer import Layer
from llm_from_scratch.core.optimizer import GradientDescent


class Trainer:
    """Train a layer using loss, gradients, and an optimizer."""

    def __init__(
        self,
        layer: Layer,
        loss_function: MeanSquaredError,
        optimizer: GradientDescent,
    ):
        self.layer = layer
        self.loss_function = loss_function
        self.optimizer = optimizer

    def train_step(
        self,
        input_value: float,
        targets: list[float],
    ) -> float:
        """
        Perform one complete training step.

        Returns:
            The loss calculated for this step.
        """

        # 1. Forward pass
        predictions = self.layer.forward(input_value)

        # 2. Calculate loss
        loss = self.loss_function.forward(
            predictions=predictions,
            targets=targets,
        )

        # 3. Calculate gradient of the loss
        loss_gradients = self.loss_function.backward(
            predictions=predictions,
            targets=targets,
        )

        # 4. Propagate gradients back through the layer
        self.layer.backward(loss_gradients)

        # 5. Update each neuron's parameters
        for neuron in self.layer.neurons:
            neuron.weight = self.optimizer.step(
                parameter=neuron.weight,
                gradient=neuron.gradient,
            )

            neuron.bias = self.optimizer.step(
                parameter=neuron.bias,
                gradient=neuron.bias_gradient,
            )

        return loss