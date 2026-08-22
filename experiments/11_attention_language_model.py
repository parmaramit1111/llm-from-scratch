"""
Experiment 11: Attention Language Model

Build a small language model using:

    Token IDs
        ↓
    Embedding
        ↓
    Self-Attention
        ↓
    Output projection
        ↓
    Softmax Cross Entropy
        ↓
    Next-token prediction

This experiment trains the attention and output projection
parameters from scratch on a tiny character-level dataset.
"""

import math

from llm_from_scratch.core.attention import SelfAttention
from llm_from_scratch.core.embedding import Embedding
from llm_from_scratch.core.softmax_cross_entropy import (
    SoftmaxCrossEntropy,
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


def matmul(
    left: list[list[float]],
    right: list[list[float]],
) -> list[list[float]]:
    """Multiply two matrices."""

    return [
        [
            sum(
                left[row][index] * right[index][column]
                for index in range(len(right))
            )
            for column in range(len(right[0]))
        ]
        for row in range(len(left))
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

    embedding = Embedding(
        vocabulary_size=vocabulary_size,
        embedding_size=embedding_size,
    )

    attention = SelfAttention(
        input_size=embedding_size,
        attention_size=attention_size,
    )

    # The flattened attention output has:
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
            # Forward
            # -------------------------------------------------

            embedded = embedding.forward(
                token_ids=token_ids,
            )

            attended = attention.forward(
                embedded,
            )

            flattened = flatten(attended)

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
            # Backward: output projection
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

            # Gradient with respect to flattened attention output.
            flattened_gradients = [
                sum(
                    output_weights[row][column]
                    * logit_gradients[column]
                    for column in range(vocabulary_size)
                )
                for row in range(output_size)
            ]

            attended_gradients = [
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
            # Backward: Self-Attention
            # -------------------------------------------------

            embedding_gradients = attention.backward(
                attended_gradients,
            )

            # -------------------------------------------------
            # Backward: Embedding
            # -------------------------------------------------

            embedding.backward(
                token_ids=token_ids,
                output_gradients=embedding_gradients,
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
            # Update Embedding
            # -------------------------------------------------

            for row in range(vocabulary_size):
                for column in range(embedding_size):
                    embedding.weights[row][column] -= (
                        learning_rate
                        * embedding.gradients[row][column]
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

        embedded = embedding.forward(
            token_ids=token_ids,
        )

        attended = attention.forward(
            embedded,
        )

        flattened = flatten(attended)

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

        embedded = embedding.forward(
            token_ids=token_ids,
        )

        attention.forward(embedded)

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