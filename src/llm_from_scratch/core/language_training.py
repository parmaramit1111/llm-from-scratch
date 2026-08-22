"""
Concept: Language Model Training

Language-model training follows:

    Forward → Logits → Loss → Backward → Update

The network produces raw logits.

SoftmaxCrossEntropy converts those logits into a loss
and calculates the gradient with respect to the logits.

When an embedding is used:

    Token IDs
        ↓
    Embedding
        ↓
    Network
        ↓
    Logits
        ↓
    Loss

During backpropagation, gradients flow back through
the network and then into the embedding.
"""

from llm_from_scratch.core.network import Network
from llm_from_scratch.core.optimizer import GradientDescent
from llm_from_scratch.core.softmax_cross_entropy import SoftmaxCrossEntropy
from llm_from_scratch.language.embedded_sequence import EmbeddedSequence


class LanguageTrainer:
    """Train a network for next-token prediction."""

    def __init__(
        self,
        network: Network,
        loss_function: SoftmaxCrossEntropy,
        optimizer: GradientDescent,
        embedded_sequence: EmbeddedSequence | None = None,
    ):
        self.network = network
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.embedded_sequence = embedded_sequence

    def train_step(
        self,
        input_values: list[float],
        target_index: int,
    ) -> float:
        """
        Perform one language-model training step.

        Returns:
            The loss calculated for this step.
        """

        # 1. Convert token IDs into network inputs when
        #    an embedding is configured.
        if self.embedded_sequence is not None:
            token_ids = [
                int(token_id)
                for token_id in input_values
            ]

            network_inputs = self.embedded_sequence.forward(
                token_ids=token_ids,
            )
        else:
            token_ids = None
            network_inputs = input_values

        # 2. Forward pass through the network.
        logits = self.network.forward(network_inputs)

        # 3. Calculate classification loss.
        loss = self.loss_function.forward(
            logits=logits,
            target_index=target_index,
        )

        # 4. Calculate gradient with respect to logits.
        logit_gradients = self.loss_function.backward(
            logits=logits,
            target_index=target_index,
        )

        # 5. Propagate gradients backward through the network.
        input_gradients = self.network.backward(
            logit_gradients,
        )

        # 6. Propagate network input gradients back into
        #    the embedding.
        if self.embedded_sequence is not None:
            self.embedded_sequence.backward(
                token_ids=token_ids,
                output_gradients=input_gradients,
            )

        # 7. Update every neuron's parameters.
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

        # 8. Update embedding parameters.
        if self.embedded_sequence is not None:
            for row_index, row in enumerate(
                self.embedded_sequence.embedding.weights
            ):
                self.embedded_sequence.embedding.weights[row_index] = [
                    self.optimizer.step(
                        parameter=weight,
                        gradient=gradient,
                    )
                    for weight, gradient in zip(
                        row,
                        self.embedded_sequence.embedding.gradients[
                            row_index
                        ],
                        strict=True,
                    )
                ]

        return loss