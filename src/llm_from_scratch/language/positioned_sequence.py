"""
Concept: Positioned Sequence

Combines token embeddings with positional embeddings.

Flow:

    Token IDs
        ↓
    Token Embedding
        ↓
    Positional Embedding
        ↓
    Position-aware token representations
"""


from llm_from_scratch.core.embedding import Embedding
from llm_from_scratch.core.positional_embedding import (
    PositionalEmbedding,
)


class PositionedSequence:
    """Combine token and positional embeddings."""

    def __init__(
        self,
        embedding: Embedding,
        positional_embedding: PositionalEmbedding,
    ):
        self.embedding = embedding
        self.positional_embedding = positional_embedding

        self.token_ids: list[int] = []

    def forward(
        self,
        token_ids: list[int],
    ) -> list[list[float]]:
        """Create position-aware token representations."""

        self.token_ids = token_ids

        token_embeddings = self.embedding.forward(
            token_ids=token_ids,
        )

        return self.positional_embedding.forward(
            token_embeddings=token_embeddings,
        )

    def backward(
        self,
        output_gradients: list[list[float]],
    ) -> list[list[float]]:
        """Propagate gradients through positional and token embeddings."""

        token_embedding_gradients = (
            self.positional_embedding.backward(
                output_gradients=output_gradients,
            )
        )

        self.embedding.backward(
            token_ids=self.token_ids,
            output_gradients=token_embedding_gradients,
        )

        return token_embedding_gradients