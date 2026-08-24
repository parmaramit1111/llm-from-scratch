from llm_from_scratch.core.activation import Activation
from llm_from_scratch.core.layer import Layer
from llm_from_scratch.core.loss import MeanSquaredError
from llm_from_scratch.core.network import Network
from llm_from_scratch.core.neuron import Neuron
from llm_from_scratch.core.optimizer import GradientDescent
from llm_from_scratch.core.training import Trainer


training_data = [
    (-2.0, [-3.0]),
    (-1.0, [-1.0]),
    (0.0, [1.0]),
    (1.0, [3.0]),
    (2.0, [5.0]),
    (3.0, [7.0]),
]


def main():
    """Run the multiple-layer training experiment."""

    layer_1 = Layer([
        Neuron(
            weights=[0.5],
            bias=0.0,
            activation=Activation(),
        ),
        Neuron(
            weights=[-0.5],
            bias=0.0,
            activation=Activation(),
        ),
    ])

    activation_layer_2 = Activation("linear")

    layer_2 = Layer([
        Neuron(
            weights=[0.5, 0.5],
            bias=0.0,
            activation=activation_layer_2,
        ),
    ])

    network = Network([
        layer_1,
        layer_2,
    ])

    trainer = Trainer(
        network=network,
        loss_function=MeanSquaredError(),
        optimizer=GradientDescent(
            learning_rate=0.01,
        ),
    )

    print()
    print("Initial parameters:")

    print()
    print("Layer 1 parameters:")

    for index, neuron in enumerate(layer_1.neurons, start=1):
        print(
            f"Neuron {index} | "
            f"Weight: {neuron.weights[0]:.6f} | "
            f"Bias: {neuron.bias:.6f}"
        )

    print()
    print("Layer 2 parameters:")

    for index, neuron in enumerate(layer_2.neurons, start=1):
        print(
            f"Neuron {index} | "
            f"Weights: {neuron.weights} | "
            f"Bias: {neuron.bias:.6f}"
        )

    print()
    print("Initial predictions:")

    for input_value, targets in training_data:
        predictions = network.forward([input_value])

        print(
            f"Input: {input_value:5.1f} | "
            f"Target: {targets[0]:5.1f} | "
            f"Prediction: {predictions[0]:8.4f}"
        )

    epochs = 1000

    print()
    print("Starting training...")

    for epoch in range(epochs):
        total_loss = 0.0

        for input_value, targets in training_data:
            total_loss += trainer.train_step(
                input_values=[input_value],
                targets=targets,
            )

        average_loss = total_loss / len(training_data)

        if (epoch + 1) % 100 == 0:
            print(
                f"Epoch {epoch + 1:4d} | "
                f"Loss: {average_loss:.6f}"
            )

    print()
    print("Training complete.")

    print()
    print("Final parameters:")

    print()
    print("Layer 1 parameters:")

    for index, neuron in enumerate(layer_1.neurons, start=1):
        print(
            f"Neuron {index} | "
            f"Weight: {neuron.weights[0]:.6f} | "
            f"Bias: {neuron.bias:.6f}"
        )

    print()
    print("Layer 2 parameters:")

    for index, neuron in enumerate(layer_2.neurons, start=1):
        print(
            f"Neuron {index} | "
            f"Weights: {neuron.weights} | "
            f"Bias: {neuron.bias:.6f}"
        )

    print()
    print("Final predictions:")

    for input_value, targets in training_data:
        predictions = network.forward([input_value])

        print(
            f"Input: {input_value:5.1f} | "
            f"Target: {targets[0]:5.1f} | "
            f"Prediction: {predictions[0]:8.4f}"
        )


if __name__ == "__main__":
    main()