"""
Concept: Positional Embedding

Positional embeddings provide information about where
a token appears in a sequence.

Token embedding:

    token_id → token vector

Positional embedding:

    position → position vector

Combined representation:

    token vector + position vector
"""


class PositionalEmbedding:
    """Add trainable positional information to token vectors."""

    def __init__(
        self,
        maximum_sequence_length: int,
        embedding_size: int,
    ):
        self.maximum_sequence_length = (
            maximum_sequence_length
        )
        self.embedding_size = embedding_size

        self.weights = [
            [0.1 for _ in range(embedding_size)]
            for _ in range(maximum_sequence_length)
        ]

        self.gradients = [
            [0.0 for _ in range(embedding_size)]
            for _ in range(maximum_sequence_length)
        ]

        self.inputs: list[list[float]] = []

    def forward(
        self,
        token_embeddings: list[list[float]],
    ) -> list[list[float]]:
        """Add positional embeddings to token embeddings."""

        self.inputs = token_embeddings

        return [
            [
                token_value + position_value
                for token_value, position_value in zip(
                    token_embedding,
                    self.weights[position],
                    strict=True,
                )
            ]
            for position, token_embedding in enumerate(
                token_embeddings
            )
        ]

    def backward(
        self,
        output_gradients: list[list[float]],
    ) -> list[list[float]]:
        """Propagate gradients through positional embeddings."""

        input_gradients = [
            row.copy()
            for row in output_gradients
        ]

        self.gradients = [
            [0.0 for _ in range(self.embedding_size)]
            for _ in range(self.maximum_sequence_length)
        ]

        for position, gradient_row in enumerate(
            output_gradients
        ):
            self.gradients[position] = gradient_row.copy()

        return input_gradients