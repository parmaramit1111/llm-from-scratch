"""
Experiment 12: Small Language Model

Train a tiny language model to predict the next character.

Training examples:

    he → l
    el → l
    ll → o
"""

from llm_from_scratch.models.small_language_model import (
    SmallLanguageModel,
)


# ---------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------

VOCABULARY = [
    "h",
    "e",
    "l",
    "o",
]

TOKEN_TO_ID = {
    token: index
    for index, token in enumerate(VOCABULARY)
}

ID_TO_TOKEN = {
    index: token
    for token, index in TOKEN_TO_ID.items()
}


# ---------------------------------------------------------
# Training data
# ---------------------------------------------------------

TRAINING_DATA = [
    ([TOKEN_TO_ID["h"], TOKEN_TO_ID["e"]], TOKEN_TO_ID["l"]),
    ([TOKEN_TO_ID["e"], TOKEN_TO_ID["l"]], TOKEN_TO_ID["l"]),
    ([TOKEN_TO_ID["l"], TOKEN_TO_ID["l"]], TOKEN_TO_ID["o"]),
]


def predict(
    model: SmallLanguageModel,
    token_ids: list[int],
) -> int:
    """Predict the most likely next token."""

    probabilities = model.forward(token_ids)

    final_probabilities = probabilities[-1]

    return max(
        range(len(final_probabilities)),
        key=lambda index: final_probabilities[index],
    )


def main() -> None:
    """Train and evaluate the small language model."""

    print("Training small language model...")
    print()

    model = SmallLanguageModel(
        vocabulary_size=len(VOCABULARY),
        embedding_size=3,
        attention_size=3,
        feed_forward_hidden_size=6,
        sequence_length=2,
    )

    epochs = 2000
    learning_rate = 0.05

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

    for epoch in range(1, epochs + 1):
        total_loss = 0.0

        for inputs, target in TRAINING_DATA:
            model.forward(inputs)

            loss = model.loss(target)

            model.backward(target)

            model.update(learning_rate)

            total_loss += loss

        if epoch % 100 == 0:
            average_loss = (
                total_loss / len(TRAINING_DATA)
            )

            print(
                f"Epoch {epoch:4d} | "
                f"Loss: {average_loss:.6f}"
            )

    print()
    print("Training complete.")
    print()

    # ---------------------------------------------------------
    # Predictions
    # ---------------------------------------------------------

    print("Predictions:")
    print()

    for inputs, target in TRAINING_DATA:
        prediction = predict(
            model,
            inputs,
        )

        probabilities = model.probabilities[-1]

        input_text = "".join(
            ID_TO_TOKEN[token_id]
            for token_id in inputs
        )

        target_token = ID_TO_TOKEN[target]
        predicted_token = ID_TO_TOKEN[prediction]

        print(
            f"Input: {input_text} | "
            f"Target: {target_token} | "
            f"Prediction: {predicted_token}"
        )

        print("Probabilities:")

        for token_id, probability in enumerate(
            probabilities
        ):
            print(
                f"  {ID_TO_TOKEN[token_id]}: "
                f"{probability:.6f}"
            )

        print()


if __name__ == "__main__":
    main()