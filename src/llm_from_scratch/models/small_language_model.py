"""
Concept: Small Language Model

A small trainable language model built from the components
developed throughout the project.

Pipeline:

    Token IDs
        ↓
    Token Embeddings
        ↓
    Positional Embeddings
        ↓
    Transformer Block
        ↓
    Output Projection
        ↓
    Logits
        ↓
    Softmax
"""

import math

from llm_from_scratch.core.transformer_block import (
    TransformerBlock,
)


class SmallLanguageModel:
    """A small trainable language model."""

    def __init__(
        self,
        vocabulary_size: int,
        embedding_size: int,
        attention_size: int,
        feed_forward_hidden_size: int,
        sequence_length: int,
    ):
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.attention_size = attention_size
        self.feed_forward_hidden_size = (
            feed_forward_hidden_size
        )
        self.sequence_length = sequence_length

        # ---------------------------------------------------------
        # Token embeddings
        #
        # Each token receives a learnable vector.
        # Non-symmetric initialization gives each token a
        # distinguishable starting representation.
        # ---------------------------------------------------------

        self.token_embeddings = [
            [
                0.05 * (token_id + 1) * (column + 1)
                for column in range(embedding_size)
            ]
            for token_id in range(vocabulary_size)
        ]

        # ---------------------------------------------------------
        # Positional embeddings
        # ---------------------------------------------------------

        self.position_embeddings = [
            [
                0.03 * (position + 1) * (column + 1)
                for column in range(embedding_size)
            ]
            for position in range(sequence_length)
        ]

        # ---------------------------------------------------------
        # Transformer Block
        # ---------------------------------------------------------

        self.transformer = TransformerBlock(
            input_size=embedding_size,
            attention_size=attention_size,
            feed_forward_hidden_size=(
                feed_forward_hidden_size
            ),
        )

        # ---------------------------------------------------------
        # Output projection
        #
        # hidden → vocabulary logits
        # ---------------------------------------------------------

        self.output_weights = [
            [
                0.05 * (row + 1) * (column + 1)
                for column in range(vocabulary_size)
            ]
            for row in range(attention_size)
        ]

        self.output_biases = [
            0.0
            for _ in range(vocabulary_size)
        ]

        # ---------------------------------------------------------
        # Forward-pass state
        # ---------------------------------------------------------

        self.token_ids: list[int] = []

        self.embeddings: list[list[float]] = []

        self.positioned_embeddings: list[list[float]] = []

        self.transformer_output: list[list[float]] = []

        self.logits: list[list[float]] = []

        self.probabilities: list[list[float]] = []

        # ---------------------------------------------------------
        # Gradient state
        # ---------------------------------------------------------

        self.output_weight_gradients: list[list[float]] = []

        self.output_bias_gradients: list[float] = []

        self.transformer_gradients: list[list[float]] = []

        self.output_gradients: list[list[float]] = []

        self.input_gradients: list[list[float]] = []

        self.token_embedding_gradients: list[list[float]] = []

        self.position_embedding_gradients: list[list[float]] = []

    def _softmax(
        self,
        values: list[float],
    ) -> list[float]:
        """Calculate a numerically stable softmax."""

        maximum = max(values)

        exponentials = [
            math.exp(value - maximum)
            for value in values
        ]

        total = sum(exponentials)

        return [
            value / total
            for value in exponentials
        ]

    def forward(
        self,
        token_ids: list[int],
    ) -> list[list[float]]:
        """Calculate next-token probabilities."""

        if not token_ids:
            raise ValueError(
                "token_ids must contain at least one token."
            )

        if len(token_ids) > self.sequence_length:
            raise ValueError(
                "token_ids cannot exceed sequence_length."
            )

        self.token_ids = token_ids

        # ---------------------------------------------------------
        # Token embeddings
        # ---------------------------------------------------------

        self.embeddings = [
            self.token_embeddings[token_id].copy()
            for token_id in token_ids
        ]

        # ---------------------------------------------------------
        # Add positional embeddings
        # ---------------------------------------------------------

        self.positioned_embeddings = [
            [
                token_value + position_value
                for token_value, position_value in zip(
                    token_embedding,
                    self.position_embeddings[position],
                    strict=True,
                )
            ]
            for position, token_embedding in enumerate(
                self.embeddings
            )
        ]

        # ---------------------------------------------------------
        # Transformer
        # ---------------------------------------------------------

        self.transformer_output = (
            self.transformer.forward(
                self.positioned_embeddings,
            )
        )

        # ---------------------------------------------------------
        # Output projection
        # ---------------------------------------------------------

        self.logits = [
            [
                sum(
                    self.transformer_output[row][index]
                    * self.output_weights[index][column]
                    for index in range(self.attention_size)
                )
                + self.output_biases[column]
                for column in range(self.vocabulary_size)
            ]
            for row in range(len(token_ids))
        ]

        # ---------------------------------------------------------
        # Softmax
        # ---------------------------------------------------------

        self.probabilities = [
            self._softmax(row)
            for row in self.logits
        ]

        return self.probabilities

    def loss(
        self,
        target_token_id: int,
    ) -> float:
        """Calculate cross-entropy loss for the final position."""

        if not self.probabilities:
            raise ValueError(
                "forward() must be called before loss()."
            )

        probability = self.probabilities[-1][target_token_id]

        return -math.log(
            max(probability, 1e-12),
        )

    def backward(
        self,
        target_token_id: int,
    ) -> list[list[float]]:
        """Propagate the loss gradient through the model."""

        if not self.probabilities:
            raise ValueError(
                "forward() must be called before backward()."
            )

        # ---------------------------------------------------------
        # Cross-entropy + softmax
        #
        # dL/dlogits = probabilities - target
        # ---------------------------------------------------------

        self.output_gradients = [
            [0.0 for _ in range(self.vocabulary_size)]
            for _ in range(len(self.logits))
        ]

        final_position = len(self.logits) - 1

        for token_id in range(self.vocabulary_size):
            self.output_gradients[final_position][token_id] = (
                self.probabilities[final_position][token_id]
            )

        self.output_gradients[final_position][
            target_token_id
        ] -= 1.0

        # ---------------------------------------------------------
        # Output projection
        #
        # logits = hidden × W + b
        # ---------------------------------------------------------

        self.output_weight_gradients = [
            [
                sum(
                    self.transformer_output[row][index]
                    * self.output_gradients[row][column]
                    for row in range(len(self.logits))
                )
                for column in range(self.vocabulary_size)
            ]
            for index in range(self.attention_size)
        ]

        self.output_bias_gradients = [
            sum(
                self.output_gradients[row][column]
                for row in range(len(self.logits))
            )
            for column in range(self.vocabulary_size)
        ]

        # ---------------------------------------------------------
        # Gradient into Transformer output
        #
        # dHidden = dLogits × Wᵀ
        # ---------------------------------------------------------

        self.transformer_gradients = [
            [
                sum(
                    self.output_gradients[row][column]
                    * self.output_weights[index][column]
                    for column in range(self.vocabulary_size)
                )
                for index in range(self.attention_size)
            ]
            for row in range(len(self.logits))
        ]

        # ---------------------------------------------------------
        # Transformer Block
        # ---------------------------------------------------------

        self.input_gradients = self.transformer.backward(
            self.transformer_gradients,
        )

        # ---------------------------------------------------------
        # Positioned Embeddings
        #
        # positioned_embedding =
        #     token_embedding + position_embedding
        #
        # Therefore the same gradient flows to both paths.
        # ---------------------------------------------------------

        self.token_embedding_gradients = [
            row.copy()
            for row in self.input_gradients
        ]

        self.position_embedding_gradients = [
            row.copy()
            for row in self.input_gradients
        ]

        return self.input_gradients

    def update(
        self,
        learning_rate: float,
    ) -> None:
        """Update model parameters using calculated gradients."""

        # ---------------------------------------------------------
        # Token embeddings
        # ---------------------------------------------------------

        for position, token_id in enumerate(self.token_ids):
            for column in range(self.embedding_size):
                self.token_embeddings[token_id][column] -= (
                    learning_rate
                    * self.token_embedding_gradients[position][
                        column
                    ]
                )

        # ---------------------------------------------------------
        # Positional embeddings
        # ---------------------------------------------------------

        for position in range(len(self.token_ids)):
            for column in range(self.embedding_size):
                self.position_embeddings[position][column] -= (
                    learning_rate
                    * self.position_embedding_gradients[position][
                        column
                    ]
                )

        # ---------------------------------------------------------
        # Output projection
        # ---------------------------------------------------------

        for row in range(self.attention_size):
            for column in range(self.vocabulary_size):
                self.output_weights[row][column] -= (
                    learning_rate
                    * self.output_weight_gradients[row][column]
                )

        for column in range(self.vocabulary_size):
            self.output_biases[column] -= (
                learning_rate
                * self.output_bias_gradients[column]
            )

        # ---------------------------------------------------------
        # Transformer parameters
        # ---------------------------------------------------------

        transformer = self.transformer

        # ---------------------------------------------------------
        # Self-Attention projection weights
        # ---------------------------------------------------------

        attention = transformer.attention

        for row in range(attention.input_size):
            for column in range(attention.attention_size):
                attention.query_weights[row][column] -= (
                    learning_rate
                    * attention.query_weight_gradients[row][column]
                )

                attention.key_weights[row][column] -= (
                    learning_rate
                    * attention.key_weight_gradients[row][column]
                )

                attention.value_weights[row][column] -= (
                    learning_rate
                    * attention.value_weight_gradients[row][column]
                )

        # ---------------------------------------------------------
        # Feed-Forward input projection
        # ---------------------------------------------------------

        feed_forward = transformer.feed_forward

        for row in range(feed_forward.input_size):
            for column in range(feed_forward.hidden_size):
                feed_forward.input_weights[row][column] -= (
                    learning_rate
                    * feed_forward.input_weight_gradients[
                        row
                    ][column]
                )

        for column in range(feed_forward.hidden_size):
            feed_forward.input_biases[column] -= (
                learning_rate
                * feed_forward.input_bias_gradients[column]
            )

        # ---------------------------------------------------------
        # Feed-Forward output projection
        # ---------------------------------------------------------

        for row in range(feed_forward.hidden_size):
            for column in range(feed_forward.output_size):
                feed_forward.output_weights[row][column] -= (
                    learning_rate
                    * feed_forward.output_weight_gradients[
                        row
                    ][column]
                )

        for column in range(feed_forward.output_size):
            feed_forward.output_biases[column] -= (
                learning_rate
                * feed_forward.output_bias_gradients[column]
            )

        # ---------------------------------------------------------
        # LayerNorm #1
        # ---------------------------------------------------------

        layer_norm = (
            transformer.attention_layer_normalization
        )

        for index in range(transformer.attention_size):
            layer_norm.gamma[index] -= (
                learning_rate
                * layer_norm.gamma_gradients[index]
            )

            layer_norm.beta[index] -= (
                learning_rate
                * layer_norm.beta_gradients[index]
            )

        # ---------------------------------------------------------
        # LayerNorm #2
        # ---------------------------------------------------------

        layer_norm = (
            transformer.feed_forward_layer_normalization
        )

        for index in range(transformer.attention_size):
            layer_norm.gamma[index] -= (
                learning_rate
                * layer_norm.gamma_gradients[index]
            )

            layer_norm.beta[index] -= (
                learning_rate
                * layer_norm.beta_gradients[index]
            )