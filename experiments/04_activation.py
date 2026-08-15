from llm_from_scratch.core.activation import Activation
from llm_from_scratch.core.layer import Layer
from llm_from_scratch.core.loss import MeanSquaredError
from llm_from_scratch.core.neuron import Neuron
from llm_from_scratch.core.optimizer import GradientDescent
from llm_from_scratch.core.training import Trainer


# Training data
#
# Target relationship:
#
#     y = ReLU(2 × x)
#
# Therefore:
#
#     x = -3 → y = 0
#     x = -2 → y = 0
#     x = -1 → y = 0
#     x =  0 → y = 0
#     x =  1 → y = 2
#     x =  2 → y = 4
#     x =  3 → y = 6

training_data = [
    (-3.0, [0.0]),
    (-2.0, [0.0]),
    (-1.0, [0.0]),
    (0.0, [0.0]),
    (1.0, [2.0]),
    (2.0, [4.0]),
    (3.0, [6.0]),
]

neuron = Neuron(
    weight=0.5,
    bias=1.0,
    activation=Activation(),
)

layer = Layer([neuron])

loss_function = MeanSquaredError()

optimizer = GradientDescent(
    learning_rate=0.01,
)

trainer = Trainer(
    layer=layer,
    loss_function=loss_function,
    optimizer=optimizer,
)

epochs = 1000

print("Starting training...")
print(
    "Initial weight:",
    f"{neuron.weight:.6f}",
)
print(
    "Initial bias:",
    f"{neuron.bias:.6f}",
)
print()

for epoch in range(epochs):
    total_loss = 0.0

    for input_value, targets in training_data:
        total_loss += trainer.train_step(
            input_value=input_value,
            targets=targets,
        )

    average_loss = total_loss / len(training_data)

    if (epoch + 1) % 100 == 0:
        print(
            f"Epoch {epoch + 1:4d} | "
            f"Loss: {average_loss:.6f} | "
            f"Weight: {neuron.weight:.6f} | "
            f"Bias: {neuron.bias:.6f}"
        )


print()
print("Training complete.")
print(f"Final weight: {neuron.weight:.6f}")
print(f"Final bias:   {neuron.bias:.6f}")
print()

print("Predictions:")

for input_value, targets in training_data:
    predictions = layer.forward(input_value)

    print(
        f"Input: {input_value:5.1f} | "
        f"Target: {targets[0]:5.1f} | "
        f"Prediction: {predictions[0]:8.4f}"
    )