from llm_from_scratch.core.activation import Activation
from llm_from_scratch.core.embedding import Embedding
from llm_from_scratch.core.layer import Layer
from llm_from_scratch.core.language_training import LanguageTrainer
from llm_from_scratch.core.network import Network
from llm_from_scratch.core.neuron import Neuron
from llm_from_scratch.core.optimizer import GradientDescent
from llm_from_scratch.core.softmax import Softmax
from llm_from_scratch.core.softmax_cross_entropy import SoftmaxCrossEntropy
from llm_from_scratch.language.embedded_sequence import EmbeddedSequence
from llm_from_scratch.language.sequence_dataset import SequenceDataset
from llm_from_scratch.language.vocabulary import Vocabulary


text = "hello"


def print_embeddings(
    embedding: Embedding,
    vocabulary: Vocabulary,
) -> None:
    """Print the current embedding vectors."""

    print()
    print("Embedding vectors:")

    for token_id, vector in enumerate(embedding.weights):
        token = vocabulary.decode(token_id)

        print(
            f"{token}: "
            f"{[round(value, 6) for value in vector]}"
        )


def main():
    """Run the embedded context prediction experiment."""

    vocabulary = Vocabulary(
        ["h", "e", "l", "o"],
    )

    dataset = SequenceDataset(
        text=text,
        vocabulary=vocabulary,
        context_size=2,
    )

    embedding = Embedding(
        vocabulary_size=4,
        embedding_size=3,
    )

    embedded_sequence = EmbeddedSequence(
        embedding=embedding,
    )

    network = Network([
        Layer([
            Neuron(
                weights=[
                    0.1,
                    -0.1,
                    0.05,
                    0.05,
                    -0.05,
                    0.1,
                ],
                bias=0.0,
                activation=Activation("linear"),
            ),
            Neuron(
                weights=[
                    -0.1,
                    0.1,
                    -0.05,
                    0.1,
                    0.05,
                    -0.1,
                ],
                bias=0.0,
                activation=Activation("linear"),
            ),
            Neuron(
                weights=[
                    0.05,
                    0.05,
                    -0.1,
                    -0.1,
                    0.1,
                    0.05,
                ],
                bias=0.0,
                activation=Activation("linear"),
            ),
            Neuron(
                weights=[
                    -0.05,
                    -0.05,
                    0.1,
                    0.05,
                    -0.1,
                    -0.05,
                ],
                bias=0.0,
                activation=Activation("linear"),
            ),
        ]),
    ])

    trainer = LanguageTrainer(
        network=network,
        loss_function=SoftmaxCrossEntropy(),
        optimizer=GradientDescent(
            learning_rate=0.1,
        ),
        embedded_sequence=embedded_sequence,
    )

    print("Training embedded context prediction model...")

    print_embeddings(
        embedding=embedding,
        vocabulary=vocabulary,
    )

    epochs = 1000

    print()
    print("Starting training...")

    for epoch in range(epochs):
        total_loss = 0.0

        for input_values, targets in dataset:
            total_loss += trainer.train_step(
                input_values=input_values,
                target_index=int(targets[0]),
            )

        average_loss = total_loss / len(dataset)

        if (epoch + 1) % 100 == 0:
            print(
                f"Epoch {epoch + 1:4d} | "
                f"Loss: {average_loss:.6f}"
            )

    print()
    print("Training complete.")

    print_embeddings(
        embedding=embedding,
        vocabulary=vocabulary,
    )

    softmax = Softmax()

    print()
    print("Predictions:")

    for input_values, targets in dataset:
        token_ids = [
            int(token_id)
            for token_id in input_values
        ]

        network_inputs = embedded_sequence.forward(
            token_ids=token_ids,
        )

        logits = network.forward(network_inputs)

        probabilities = softmax.forward(logits)

        predicted_index = probabilities.index(
            max(probabilities)
        )

        input_tokens = [
            vocabulary.decode(token_id)
            for token_id in token_ids
        ]

        input_text = "".join(input_tokens)

        target_token = vocabulary.decode(
            int(targets[0])
        )

        predicted_token = vocabulary.decode(
            predicted_index
        )

        print()
        print(
            f"Input: {input_text} | "
            f"Target: {target_token} | "
            f"Prediction: {predicted_token}"
        )

        print("Probabilities:")

        for index, probability in enumerate(probabilities):
            token = vocabulary.decode(index)

            print(
                f"  {token}: {probability:.6f}"
            )


if __name__ == "__main__":
    main()