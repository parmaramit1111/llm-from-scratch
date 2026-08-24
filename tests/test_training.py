"""
Tests for the Trainer.

The training step should:

    1. Make predictions through the network
    2. Calculate the loss
    3. Calculate gradients
    4. Propagate gradients backward through the network
    5. Update weights and biases

The important result is that the loss should decrease
after a training step.
"""

from llm_from_scratch.core.network import Network
from llm_from_scratch.core.layer import Layer
from llm_from_scratch.core.loss import MeanSquaredError
from llm_from_scratch.core.neuron import Neuron
from llm_from_scratch.core.optimizer import GradientDescent
from llm_from_scratch.core.training import Trainer


def test_training_step_reduces_loss():
    """A training step should move the network toward the targets."""

    neurons = [
        Neuron(weights=[2.0]),
        Neuron(weights=[2.0]),
        Neuron(weights=[2.0]),
    ]

    layer = Layer(neurons)

    network = Network([
        layer,
    ])

    loss_function = MeanSquaredError()
    optimizer = GradientDescent(learning_rate=0.01)

    trainer = Trainer(
        network=network,
        loss_function=loss_function,
        optimizer=optimizer,
    )

    input_values = [3.0]
    targets = [9.0, 9.0, 9.0]

    initial_predictions = layer.forward(input_values)

    initial_loss = loss_function.forward(
        predictions=initial_predictions,
        targets=targets,
    )

    trainer.train_step(
        input_values=input_values,
        targets=targets,
    )

    updated_predictions = layer.forward(input_values)

    updated_loss = loss_function.forward(
        predictions=updated_predictions,
        targets=targets,
    )

    assert updated_loss < initial_loss


def test_training_step_updates_weight_and_bias():
    """A training step should update weights and biases of all neurons."""

    neurons = [
        Neuron(weights=[1.0], bias=0.0),
        Neuron(weights=[2.0], bias=0.0),
    ]

    targets = [1.0, 2.0]

    layer = Layer(neurons)
    network = Network([
        layer,
    ])
    loss_function = MeanSquaredError()
    optimizer = GradientDescent(learning_rate=0.01)

    trainer = Trainer(
        network=network,
        loss_function=loss_function,
        optimizer=optimizer,
    )

    initial_weights = [neuron.weights for neuron in layer.neurons]
    initial_biases = [neuron.bias for neuron in layer.neurons]

    trainer.train_step(
        input_values=[2.0],
        targets=targets,
    )

    updated_weights = [neuron.weights for neuron in layer.neurons]
    updated_biases = [neuron.bias for neuron in layer.neurons]

    assert updated_weights != initial_weights
    assert updated_biases != initial_biases

def test_training_step_updates_multiple_layers():
    """A training step should update weights and biases in every layer."""

    layer_1 = Layer([
        Neuron(weights=[1.0, 0.5], bias=0.0),
        Neuron(weights=[0.5, 1.0], bias=0.0),
    ])

    layer_2 = Layer([
        Neuron(weights=[1.0, 1.0], bias=0.0),
    ])

    network = Network([
        layer_1,
        layer_2,
    ])

    loss_function = MeanSquaredError()
    optimizer = GradientDescent(learning_rate=0.01)

    trainer = Trainer(
        network=network,
        loss_function=loss_function,
        optimizer=optimizer,
    )

    initial_layer_1_weights = [
        neuron.weights.copy()
        for neuron in layer_1.neurons
    ]
    initial_layer_1_biases = [
        neuron.bias
        for neuron in layer_1.neurons
    ]

    initial_layer_2_weights = [
        neuron.weights.copy()
        for neuron in layer_2.neurons
    ]
    initial_layer_2_biases = [
        neuron.bias
        for neuron in layer_2.neurons
    ]

    trainer.train_step(
        input_values=[2.0, 3.0],
        targets=[10.0],
    )

    updated_layer_1_weights = [
        neuron.weights
        for neuron in layer_1.neurons
    ]
    updated_layer_1_biases = [
        neuron.bias
        for neuron in layer_1.neurons
    ]

    updated_layer_2_weights = [
        neuron.weights
        for neuron in layer_2.neurons
    ]
    updated_layer_2_biases = [
        neuron.bias
        for neuron in layer_2.neurons
    ]

    assert updated_layer_1_weights != initial_layer_1_weights
    assert updated_layer_1_biases != initial_layer_1_biases

    assert updated_layer_2_weights != initial_layer_2_weights
    assert updated_layer_2_biases != initial_layer_2_biases