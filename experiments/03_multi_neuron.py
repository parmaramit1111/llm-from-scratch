from llm_from_scratch.core.layer import Layer
from llm_from_scratch.core.loss import MeanSquaredError
from llm_from_scratch.core.neuron import Neuron
from llm_from_scratch.core.optimizer import GradientDescent
from llm_from_scratch.core.training import Trainer


# Training data
training_data = [
    (1.0, [3.0, 5.0, 7.0]),
    (2.0, [6.0, 10.0, 14.0]),
    (3.0, [9.0, 15.0, 21.0]),
]


# Create a layer with three neurons
neurons = [
    Neuron(weight=0.5),
    Neuron(weight=0.5),
    Neuron(weight=0.5),
]

layer = Layer(neurons)

loss_function = MeanSquaredError()
optimizer = GradientDescent(learning_rate=0.01)

trainer = Trainer(
    layer=layer,
    loss_function=loss_function,
    optimizer=optimizer,
)


# Training
epochs = 1000

print("Starting training...")
print(f"Initial weights: {[neuron.weight for neuron in layer.neurons]}")
print()

for epoch in range(epochs):
    total_loss = 0.0

    for input_value, targets in training_data:
        total_loss += trainer.train_step(
            input_value=input_value,
            targets=targets,
        )

    average_loss = total_loss / len(training_data)

    if (epoch + 1) % 10 == 0:
        weights = [
            neuron.weight
            for neuron in layer.neurons
        ]

        print(
            f"Epoch {epoch + 1:3d} | "
            f"Loss: {average_loss:.6f} | "
            f"Weights: {weights}"
        )


print()
print("Training complete.")
print(
    "Final weights:",
    [neuron.weight for neuron in layer.neurons],
)