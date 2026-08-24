"""
Concept: Softmax

Softmax converts raw model outputs (logits) into probabilities.

For logits:

    [z1, z2, ..., zn]

Softmax:

    softmax(zi) = exp(zi) / sum(exp(zj))

The resulting values:

- Are between 0 and 1
- Sum to 1
- Represent a probability distribution

For numerical stability, we subtract the maximum logit
before calculating the exponentials.
"""

import math


class Softmax:
    """Convert logits into a probability distribution."""

    def forward(
        self,
        logits: list[float],
    ) -> list[float]:
        """Convert logits into probabilities."""

        maximum = max(logits)

        exponentials = [
            math.exp(logit - maximum)
            for logit in logits
        ]

        total = sum(exponentials)

        return [
            value / total
            for value in exponentials
        ]