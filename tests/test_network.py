from llm_from_scratch.core.layer import Layer
from llm_from_scratch.core.network import Network
from llm_from_scratch.core.neuron import Neuron


def test_network_forwards_through_multiple_layers():
    """A network should pass each layer's output to the next layer."""

    layer_1 = Layer([
        Neuron(weights=[2.0], bias=1.0),
        Neuron(weights=[3.0], bias=0.0),
    ])

    layer_2 = Layer([
        Neuron(weights=[2.0, 3.0], bias=1.0),
    ])

    network = Network([
        layer_1,
        layer_2,
    ])

    output = network.forward([4.0])

    assert output == [55.0]

def test_network_backwards_through_multiple_layers():
    """A network should propagate gradients through layers in reverse order."""

    layer_1 = Layer([
        Neuron(weights=[2.0], bias=1.0),
        Neuron(weights=[3.0], bias=0.0),
    ])

    layer_2 = Layer([
        Neuron(weights=[2.0, 3.0], bias=1.0),
    ])

    network = Network([
        layer_1,
        layer_2,
    ])

    network.forward([4.0])

    input_gradients = network.backward(
        output_gradients=[1.0],
    )

    assert input_gradients == [13.0]