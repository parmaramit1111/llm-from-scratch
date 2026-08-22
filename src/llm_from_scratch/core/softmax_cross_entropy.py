"""
Concept: Softmax + Cross-Entropy

Softmax converts logits into probabilities.

Cross-Entropy measures how much probability
the model assigned to the correct class.

For a target class:

    loss = -log(probability_of_target)

When Softmax and Cross-Entropy are combined,
the gradient with respect to the logits simplifies to:

    gradient = probabilities - one_hot(target)
"""

from llm_from_scratch.core.loss import CrossEntropy
from llm_from_scratch.core.softmax import Softmax


class SoftmaxCrossEntropy:
    """Calculate classification loss and logits gradients."""

    def __init__(self):
        self.softmax = Softmax()
        self.cross_entropy = CrossEntropy()

    def forward(
        self,
        logits: list[float],
        target_index: int,
    ) -> float:
        """Calculate cross-entropy loss from logits."""

        probabilities = self.softmax.forward(logits)

        return self.cross_entropy.forward(
            probabilities=probabilities,
            target_index=target_index,
        )

    def backward(
        self,
        logits: list[float],
        target_index: int,
    ) -> list[float]:
        """Calculate gradient of loss with respect to logits."""

        probabilities = self.softmax.forward(logits)

        gradients = probabilities.copy()
        gradients[target_index] -= 1.0

        return gradients