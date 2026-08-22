"""
Concept: Embedded Sequence

An embedded sequence converts token IDs into embedding vectors
and flattens those vectors so they can be passed to the existing
feed-forward network.

Example:

    token IDs:
        [1, 2]

    embedding:
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ]

    flattened:
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

During backward propagation, the flattened gradients are
reshaped and passed back to the embedding layer.
"""

from llm_from_scratch.core.embedding import Embedding


class EmbeddedSequence:
    """Convert token IDs into flattened embedding vectors."""

    def __init__(
        self,
        embedding: Embedding,
    ):
        self.embedding = embedding

    def forward(
        self,
        token_ids: list[int],
    ) -> list[float]:
        """Look up and flatten embedding vectors."""

        vectors = self.embedding.forward(token_ids)

        return [
            value
            for vector in vectors
            for value in vector
        ]

    def backward(
        self,
        token_ids: list[int],
        output_gradients: list[float],
    ) -> None:
        """Reshape flattened gradients and update embedding gradients."""

        embedding_size = self.embedding.embedding_size

        gradients = [
            output_gradients[index:index + embedding_size]
            for index in range(
                0,
                len(output_gradients),
                embedding_size,
            )
        ]

        self.embedding.backward(
            token_ids=token_ids,
            output_gradients=gradients,
        )