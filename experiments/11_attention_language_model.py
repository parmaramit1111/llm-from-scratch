"""
Experiment 11: Attention Language Model

Build a small language model using:

    Token IDs
        ↓
    Token Embedding
        ↓
    Positional Embedding
        ↓
    Self-Attention
        ↓
    Feed-Forward Network
        ↓
    Output projection
        ↓
    Softmax Cross Entropy
        ↓
    Next-token prediction

This experiment trains token embeddings, positional embeddings,
self-attention parameters, feed-forward parameters, and the
output projection from scratch on a tiny character-level dataset.
"""

import math

from llm_from_scratch.core.attention import SelfAttention
from llm_from_scratch.core.embedding import Embedding
from llm_from_scratch.core.feed_forward import FeedForward
from llm_from_scratch.core.positional_embedding import (
    PositionalEmbedding,
)
from llm_from_scratch.core.softmax_cross_entropy import (
    SoftmaxCrossEntropy,
)
from llm_from_scratch.language.positioned_sequence import (
    PositionedSequence,
)


VOCABULARY = ["h", "e", "l", "o"]

TOKEN_TO_ID = {
    token: index
    for index, token in enumerate(VOCABULARY)
}

TRAINING_EXAMPLES = [
    ("he", "l"),
    ("el", "l"),
    ("ll", "o"),
]


def flatten(
    values: list[list[float]],
) -> list[float]:
    """Flatten a matrix into one vector."""

    return [
        value
        for row in values
        for value in row
    ]


def softmax(
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


def main() -> None:
    """Train and evaluate the attention language model."""

    print("Training attention language model...")
    print()

    vocabulary_size = len(VOCABULARY)
    embedding_size = 3
    attention_size = 3
    feed_forward_hidden_size = 6

    embedding = Embedding(
        vocabulary_size=vocabulary_size,
        embedding_size=embedding_size,
    )

    positioned_sequence = PositionedSequence(
        embedding=embedding,
        positional_embedding=PositionalEmbedding(
            embedding_size=embedding_size,
            maximum_sequence_length=2,
        ),
    )

    attention = SelfAttention(
        input_size=embedding_size,
        attention_size=attention_size,
    )

    feed_forward = FeedForward(
        input_size=attention_size,
        hidden_size=feed_forward_hidden_size,
        output_size=attention_size,
    )

    # The flattened feed-forward output has:
    #
    #     sequence_length × attention_size
    #
    # For our two-character inputs:
    #
    #     2 × 3 = 6
    #
    output_size = 2 * attention_size

    output_weights = [
        [0.1 for _ in range(vocabulary_size)]
        for _ in range(output_size)
    ]

    output_biases = [
        0.0
        for _ in range(vocabulary_size)
    ]

    loss_function = SoftmaxCrossEntropy()

    learning_rate = 0.1
    epochs = 1000

    for epoch in range(1, epochs + 1):
        total_loss = 0.0

        for input_text, target_token in TRAINING_EXAMPLES:
            token_ids = [
                TOKEN_TO_ID[token]
                for token in input_text
            ]

            target_index = TOKEN_TO_ID[target_token]

            # -------------------------------------------------
            # Forward: Embedding + Position
            # -------------------------------------------------

            positioned = positioned_sequence.forward(
                token_ids=token_ids,
            )

            # -------------------------------------------------
            # Forward: Self-Attention
            # -------------------------------------------------

            attended = attention.forward(
                positioned,
            )

            # -------------------------------------------------
            # Forward: Feed-Forward Network
            # -------------------------------------------------

            transformed = feed_forward.forward(
                attended,
            )

            flattened = flatten(transformed)

            logits = [
                sum(
                    flattened[index] * output_weights[index][column]
                    for index in range(output_size)
                )
                + output_biases[column]
                for column in range(vocabulary_size)
            ]

            loss = loss_function.forward(
                logits=logits,
                target_index=target_index,
            )

            total_loss += loss

            # -------------------------------------------------
            # Backward: Output projection
            # -------------------------------------------------

            logit_gradients = loss_function.backward(
                logits=logits,
                target_index=target_index,
            )

            output_weight_gradients = [
                [
                    flattened[row] * logit_gradients[column]
                    for column in range(vocabulary_size)
                ]
                for row in range(output_size)
            ]

            output_bias_gradients = logit_gradients

            # Gradient with respect to flattened FFN output.
            flattened_gradients = [
                sum(
                    output_weights[row][column]
                    * logit_gradients[column]
                    for column in range(vocabulary_size)
                )
                for row in range(output_size)
            ]

            transformed_gradients = [
                flattened_gradients[
                    start:start + attention_size
                ]
                for start in range(
                    0,
                    output_size,
                    attention_size,
                )
            ]

            # -------------------------------------------------
            # Backward: Feed-Forward Network
            # -------------------------------------------------

            attended_gradients = feed_forward.backward(
                transformed_gradients,
            )

            # -------------------------------------------------
            # Backward: Self-Attention
            # -------------------------------------------------

            positioned_gradients = attention.backward(
                attended_gradients,
            )

            # -------------------------------------------------
            # Backward: Positioned Sequence
            # -------------------------------------------------

            positioned_sequence.backward(
                output_gradients=positioned_gradients,
            )

            # -------------------------------------------------
            # Update output projection
            # -------------------------------------------------

            for row in range(output_size):
                for column in range(vocabulary_size):
                    output_weights[row][column] -= (
                        learning_rate
                        * output_weight_gradients[row][column]
                    )

            for column in range(vocabulary_size):
                output_biases[column] -= (
                    learning_rate
                    * output_bias_gradients[column]
                )

            # -------------------------------------------------
            # Update Self-Attention
            # -------------------------------------------------

            for row in range(embedding_size):
                for column in range(attention_size):
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

            # -------------------------------------------------
            # Update Feed-Forward Network
            # -------------------------------------------------

            for row in range(attention_size):
                for column in range(feed_forward_hidden_size):
                    feed_forward.input_weights[row][column] -= (
                        learning_rate
                        * feed_forward.input_weight_gradients[row][column]
                    )

            for column in range(feed_forward_hidden_size):
                feed_forward.input_biases[column] -= (
                    learning_rate
                    * feed_forward.input_bias_gradients[column]
                )

            for row in range(feed_forward_hidden_size):
                for column in range(attention_size):
                    feed_forward.output_weights[row][column] -= (
                        learning_rate
                        * feed_forward.output_weight_gradients[row][column]
                    )

            for column in range(attention_size):
                feed_forward.output_biases[column] -= (
                    learning_rate
                    * feed_forward.output_bias_gradients[column]
                )

            # -------------------------------------------------
            # Update Token Embedding
            # -------------------------------------------------

            for row in range(vocabulary_size):
                for column in range(embedding_size):
                    embedding.weights[row][column] -= (
                        learning_rate
                        * embedding.gradients[row][column]
                    )

            # -------------------------------------------------
            # Update Positional Embedding
            # -------------------------------------------------

            for row in range(
                positioned_sequence.positional_embedding.maximum_sequence_length
            ):
                for column in range(embedding_size):
                    positioned_sequence.positional_embedding.weights[
                        row
                    ][column] -= (
                        learning_rate
                        * positioned_sequence.positional_embedding.gradients[
                            row
                        ][column]
                    )

        if epoch % 100 == 0:
            average_loss = (
                total_loss / len(TRAINING_EXAMPLES)
            )

            print(
                f"Epoch {epoch:4d} | "
                f"Loss: {average_loss:.6f}"
            )

    print()
    print("Training complete.")
    print()
    print("Predictions:")
    print()

    # ---------------------------------------------------------
    # Predictions
    # ---------------------------------------------------------

    for input_text, target_token in TRAINING_EXAMPLES:
        token_ids = [
            TOKEN_TO_ID[token]
            for token in input_text
        ]

        positioned = positioned_sequence.forward(
            token_ids=token_ids,
        )

        attended = attention.forward(
            positioned,
        )

        transformed = feed_forward.forward(
            attended,
        )

        flattened = flatten(transformed)

        logits = [
            sum(
                flattened[index] * output_weights[index][column]
                for index in range(output_size)
            )
            + output_biases[column]
            for column in range(vocabulary_size)
        ]

        probabilities = softmax(logits)

        prediction_index = max(
            range(vocabulary_size),
            key=lambda index: probabilities[index],
        )

        prediction = VOCABULARY[prediction_index]

        print(
            f"Input: {input_text} | "
            f"Target: {target_token} | "
            f"Prediction: {prediction}"
        )

        print()
        print("Probabilities:")

        for token, probability in zip(
            VOCABULARY,
            probabilities,
            strict=True,
        ):
            print(
                f"  {token}: {probability:.6f}"
            )

        print()

    # ---------------------------------------------------------
    # Learned attention patterns
    # ---------------------------------------------------------

    print("Learned attention patterns:")
    print()

    for input_text, _ in TRAINING_EXAMPLES:
        token_ids = [
            TOKEN_TO_ID[token]
            for token in input_text
        ]

        positioned = positioned_sequence.forward(
            token_ids=token_ids,
        )

        attention.forward(
            positioned,
        )

        print(f"Input: {input_text}")
        print()

        print("        " + " ".join(
            f"{token:>8}"
            for token in input_text
        ))

        for token, row in zip(
            input_text,
            attention.attention_weights,
            strict=True,
        ):
            values = " ".join(
                f"{value:8.6f}"
                for value in row
            )

            print(f"{token}:   {values}")

        print()

        for token, row in zip(
            input_text,
            attention.attention_weights,
            strict=True,
        ):
            print(
                f"  {token} row sum: "
                f"{sum(row):.6f}"
            )

        print()


if __name__ == "__main__":
    main()