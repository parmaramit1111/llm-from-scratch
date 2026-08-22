"""
Concept: Embedding

An embedding converts token IDs into learnable vectors.

Each token has one row in the embedding matrix.

For example:

    token ID 2
        ↓
    embedding matrix row 2
        ↓
    [0.6, 0.2, -0.3]

The embedding vectors are trainable parameters.
"""

import random


class Embedding:
    """Map token IDs to learnable embedding vectors."""

    def __init__(
        self,
        vocabulary_size: int,
        embedding_size: int,
    ):
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size

        self.weights = [
            [
                random.uniform(-0.1, 0.1)
                for _ in range(embedding_size)
            ]
            for _ in range(vocabulary_size)
        ]

        self.gradients = [
            [0.0] * embedding_size
            for _ in range(vocabulary_size)
        ]

    def forward(
        self,
        token_ids: list[int],
    ) -> list[list[float]]:
        """Look up the embedding vector for each token ID."""

        return [
            self.weights[token_id].copy()
            for token_id in token_ids
        ]

    def backward(
        self,
        token_ids: list[int],
        output_gradients: list[list[float]],
    ) -> None:
        """Accumulate gradients for the used embedding rows."""

        self.gradients = [
            [0.0] * self.embedding_size
            for _ in range(self.vocabulary_size)
        ]

        for token_id, output_gradient in zip(
            token_ids,
            output_gradients,
            strict=True,
        ):
            self.gradients[token_id] = [
                gradient + output_gradient
                for gradient, output_gradient in zip(
                    self.gradients[token_id],
                    output_gradient,
                    strict=True,
                )
            ]