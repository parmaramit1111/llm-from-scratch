"""
Tests for the Trainer.

The training step should:

    1. Make a prediction
    2. Calculate the loss
    3. Calculate gradients
    4. Update the weight

The important result is that the loss should decrease
after a training step.
"""

from llm_from_scratch.core.layer import Layer
from llm_from_scratch.core.loss import MeanSquaredError
from llm_from_scratch.core.neuron import Neuron
from llm_from_scratch.core.optimizer import GradientDescent
from llm_from_scratch.core.training import Trainer


def test_training_step_reduces_loss():
    """A training step should move the layer toward the targets."""

    neurons = [
        Neuron(weight=2.0),
        Neuron(weight=2.0),
        Neuron(weight=2.0),
    ]

    layer = Layer(neurons)
    loss_function = MeanSquaredError()
    optimizer = GradientDescent(learning_rate=0.01)

    trainer = Trainer(
        layer=layer,
        loss_function=loss_function,
        optimizer=optimizer,
    )

    input_value = 3.0
    targets = [9.0, 9.0, 9.0]

    initial_predictions = layer.forward(input_value)

    initial_loss = loss_function.forward(
        predictions=initial_predictions,
        targets=targets,
    )

    trainer.train_step(
        input_value=input_value,
        targets=targets,
    )

    updated_predictions = layer.forward(input_value)

    updated_loss = loss_function.forward(
        predictions=updated_predictions,
        targets=targets,
    )

    assert updated_loss < initial_loss


def test_training_step_updates_weight_and_bias():
    """A training step should update weights and biases of all neurons."""

    neurons = [
        Neuron(weight=1.0, bias=0.0),
        Neuron(weight=2.0, bias=0.0),
    ]

    targets = [1.0, 2.0]

    layer = Layer(neurons)
    loss_function = MeanSquaredError()
    optimizer = GradientDescent(learning_rate=0.01)

    trainer = Trainer(
        layer=layer,
        loss_function=loss_function,
        optimizer=optimizer,
    )

    initial_weights = [neuron.weight for neuron in layer.neurons]
    initial_biases = [neuron.bias for neuron in layer.neurons]

    trainer.train_step(
        input_value=2.0,
        targets=targets,
    )

    updated_weights = [neuron.weight for neuron in layer.neurons]
    updated_biases = [neuron.bias for neuron in layer.neurons]

    assert updated_weights != initial_weights
    assert updated_biases != initial_biases