from llm_from_scratch.core.layer import Layer
from llm_from_scratch.core.neuron import Neuron


def test_layer_forwards_input_to_all_neurons():
    """A layer should pass the same input vector through every neuron."""
    neurons = [
        Neuron(weights=[2.0], bias=1.0),
        Neuron(weights=[3.0], bias=0.0),
        Neuron(weights=[0.5], bias=2.0),
    ]

    layer = Layer(neurons)

    outputs = layer.forward([4.0])

    assert outputs == [9.0, 12.0, 4.0]

def test_layer_backward_passes_gradients_to_neurons():
    """A layer should pass gradients to neurons and aggregate input gradients."""
    neurons = [
        Neuron(weights=[2.0], bias=1.0),
        Neuron(weights=[3.0], bias=0.0),
        Neuron(weights=[0.5], bias=2.0),
    ]

    layer = Layer(neurons)

    layer.forward([4.0])

    input_gradients = layer.backward(
        output_gradients=[-2.0, 3.0, 1.5],
    )

    # dLoss/dInput = sum of input gradients from all neurons
    assert input_gradients == [5.75]

    assert neurons[0].gradient == [-8.0]
    assert neurons[0].bias_gradient == -2.0

    assert neurons[1].gradient == [12.0]
    assert neurons[1].bias_gradient == 3.0

    assert neurons[2].gradient == [6.0]
    assert neurons[2].bias_gradient == 1.5

def test_layer_with_multiple_inputs():
    """A layer should pass multiple inputs through every neuron."""
    neurons = [
        Neuron(weights=[2.0, 3.0], bias=1.0),
        Neuron(weights=[1.0, 2.0], bias=0.0),
    ]

    layer = Layer(neurons)

    outputs = layer.forward([4.0, 5.0])

    assert outputs == [24.0, 14.0]

def test_layer_backward_aggregates_input_gradients():
    """A layer should aggregate input gradients from all neurons."""
    neurons = [
        Neuron(weights=[2.0, 3.0], bias=1.0),
        Neuron(weights=[1.0, 4.0], bias=0.0),
    ]

    layer = Layer(neurons)

    layer.forward([4.0, 5.0])

    input_gradients = layer.backward(
        output_gradients=[2.0, 3.0],
    )

    assert input_gradients == [7.0, 18.0]