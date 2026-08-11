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

from llm_from_scratch.core.loss import MeanSquaredError
from llm_from_scratch.core.neuron import Neuron
from llm_from_scratch.core.optimizer import GradientDescent
from llm_from_scratch.core.training import Trainer


def test_training_step_reduces_loss():
    """A training step should move the neuron toward the target."""

    neuron = Neuron(weight=2.0)
    loss_function = MeanSquaredError()
    optimizer = GradientDescent(learning_rate=0.01)

    trainer = Trainer(
        neuron=neuron,
        loss_function=loss_function,
        optimizer=optimizer,
    )

    input_value = 3.0
    target = 9.0

    initial_prediction = neuron.forward(input_value)
    initial_loss = loss_function.forward(
        prediction=initial_prediction,
        target=target,
    )

    trainer.train_step(
        input_value=input_value,
        target=target,
    )

    updated_prediction = neuron.forward(input_value)
    updated_loss = loss_function.forward(
        prediction=updated_prediction,
        target=target,
    )

    assert updated_loss < initial_loss


def test_training_step_updates_weight_and_bias():
    """A training step should update both weight and bias."""

    neuron = Neuron(weight=1.0, bias=0.0)
    loss_function = MeanSquaredError()
    optimizer = GradientDescent(learning_rate=0.01)

    trainer = Trainer(
        neuron=neuron,
        loss_function=loss_function,
        optimizer=optimizer,
    )

    initial_weight = neuron.weight
    initial_bias = neuron.bias

    trainer.train_step(
        input_value=2.0,
        target=8.0,
    )

    assert neuron.weight != initial_weight
    assert neuron.bias != initial_bias