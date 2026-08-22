"""
Experiment 10: Self-Attention

This experiment demonstrates how self-attention allows
each token to attend to every other token in a sequence.

Flow:

    Token embeddings
        ↓
    Self-Attention
        ↓
    Attention weights
        ↓
    Context-aware representations

This experiment does not train the model.
It focuses on understanding the forward pass.
"""

from llm_from_scratch.core.attention import SelfAttention


def print_matrix(
    matrix: list[list[float]],
    labels: list[str] | None = None,
) -> None:
    """Print a matrix with optional row labels."""

    for index, row in enumerate(matrix):
        label = (
            labels[index]
            if labels is not None
            else str(index)
        )

        values = " ".join(
            f"{value:.6f}"
            for value in row
        )

        print(f"{label}: [{values}]")


def main() -> None:
    """Run the self-attention experiment."""

    print("Self-Attention experiment...")
    print()

    # Small embedding vectors for our vocabulary:
    #
    #     h, e, l, o
    #
    # Each token is represented by a 3-dimensional vector.
    embeddings = {
        "h": [0.3, 0.1, 0.2],
        "e": [0.2, 0.4, 0.1],
        "l": [0.1, 0.3, 0.5],
        "o": [0.4, 0.2, 0.3],
    }

    # Example sequence: "hello"
    tokens = ["h", "e", "l", "l", "o"]

    inputs = [
        embeddings[token]
        for token in tokens
    ]

    print("Input sequence:")
    print(f"  {''.join(tokens)}")
    print()

    print("Input embeddings:")
    print_matrix(
        inputs,
        labels=tokens,
    )
    print()

    # Create the self-attention layer.
    attention = SelfAttention(
        input_size=3,
        attention_size=3,
    )

    # Forward pass.
    outputs = attention.forward(inputs)

    print("Attention weights:")
    print()

    print("        " + " ".join(
        f"{token:>8}"
        for token in tokens
    ))

    for token, row in zip(
        tokens,
        attention.attention_weights,
        strict=True,
    ):
        values = " ".join(
            f"{value:8.6f}"
            for value in row
        )

        print(f"{token}:   {values}")

    print()

    print("Attention output:")
    print_matrix(
        outputs,
        labels=tokens,
    )
    print()

    print("Attention row sums:")

    for token, row in zip(
        tokens,
        attention.attention_weights,
        strict=True,
    ):
        print(
            f"  {token}: "
            f"{sum(row):.6f}"
        )

    print()

    print("Experiment complete.")


if __name__ == "__main__":
    main()