"""
Concept: Training

Training repeatedly runs the learning cycle:

    Forward → Loss → Backward → Update

Each iteration gives the model an opportunity
to reduce its prediction error.
"""

from llm_from_scratch.core.loss import MeanSquaredError
from llm_from_scratch.core.network import Network
from llm_from_scratch.core.optimizer import GradientDescent


class Trainer:
    """Train a network using loss, gradients, and an optimizer."""

    def __init__(
        self,
        network: Network,
        loss_function: MeanSquaredError,
        optimizer: GradientDescent,
    ):
        self.network = network
        self.loss_function = loss_function
        self.optimizer = optimizer

    def train_step(
        self,
        input_values: list[float],
        targets: list[float],
    ) -> float:
        """
        Perform one complete training step.

        Returns:
            The loss calculated for this step.
        """

        # 1. Forward pass through the network
        predictions = self.network.forward(input_values)

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

        # 4. Propagate gradients backward through the network
        self.network.backward(loss_gradients)

        # 5. Update every neuron's parameters
        for layer in self.network.layers:
            for neuron in layer.neurons:
                neuron.weights = [
                    self.optimizer.step(weight, gradient)
                    for weight, gradient in zip(
                        neuron.weights,
                        neuron.gradient,
                        strict=True,
                    )
                ]

                neuron.bias = self.optimizer.step(
                    parameter=neuron.bias,
                    gradient=neuron.bias_gradient,
                )

        return loss