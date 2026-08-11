"""
Experiment 01: Linear Learning

Goal:
    Teach a single neuron the relationship:

        y = 3 × x

Training data:

    1 → 3
    2 → 6
    3 → 9
    4 → 12
    5 → 15

The model starts with an incorrect weight and gradually
adjusts it through training.
"""

from llm_from_scratch.core.loss import MeanSquaredError
from llm_from_scratch.core.neuron import Neuron
from llm_from_scratch.core.optimizer import GradientDescent
from llm_from_scratch.core.training import Trainer


def main() -> None:
    """Run the linear learning experiment."""

    # Training data
    training_data = [
        (1.0, 3.0),
        (2.0, 6.0),
        (3.0, 9.0),
        (4.0, 12.0),
        (5.0, 15.0),
    ]

    # Start with an incorrect weight.
    neuron = Neuron(weight=0.5)

    loss_function = MeanSquaredError()

    optimizer = GradientDescent(
        learning_rate=0.01,
    )

    trainer = Trainer(
        neuron=neuron,
        loss_function=loss_function,
        optimizer=optimizer,
    )

    epochs = 100

    print("Starting training...")
    print(f"Initial weight: {neuron.weight:.4f}")
    print()

    for epoch in range(epochs):

        total_loss = 0.0

        for input_value, target in training_data:

            loss = trainer.train_step(
                input_value=input_value,
                target=target,
            )

            total_loss += loss

        average_loss = total_loss / len(training_data)

        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:3d} | "
                f"Loss: {average_loss:.6f} | "
                f"Weight: {neuron.weight:.6f}"
            )

    print()
    print("Training complete.")
    print(f"Final weight: {neuron.weight:.6f}")


if __name__ == "__main__":
    main()