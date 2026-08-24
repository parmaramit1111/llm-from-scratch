from llm_from_scratch.core.gradient_check import GradientChecker
from llm_from_scratch.core.layer import Layer
from llm_from_scratch.core.loss import MeanSquaredError
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


def test_network_input_gradient():
    """A network input gradient should match the numerical gradient."""
    inputs = [2.0, 3.0]
    targets = [6.0]

    loss_function = MeanSquaredError()

    network = Network([
        Layer([
            Neuron(weights=[2.0, 3.0], bias=1.0),
            Neuron(weights=[4.0, 5.0], bias=0.0),
        ]),
        Layer([
            Neuron(weights=[2.0, 3.0], bias=1.0),
        ]),
    ])

    predictions = network.forward(inputs)

    loss_gradients = loss_function.backward(
        predictions=predictions,
        targets=targets,
    )

    input_gradients = network.backward(
        output_gradients=loss_gradients,
    )

    def loss_for_input(value: float) -> float:
        test_inputs = [value, inputs[1]]

        predictions = network.forward(test_inputs)

        return loss_function.forward(
            predictions=predictions,
            targets=targets,
        )

    checker = GradientChecker(
        tolerance=1e-5,
    )

    numerical_gradient = checker.numerical_gradient(
        parameter=inputs[0],
        loss_function=loss_for_input,
    )

    assert checker.check(
        analytical_gradient=input_gradients[0],
        numerical_gradient=numerical_gradient,
    )

def test_network_second_input_gradient():
    """A network second input gradient should match the numerical gradient."""
    inputs = [2.0, 3.0]
    targets = [6.0]

    loss_function = MeanSquaredError()

    network = Network([
        Layer([
            Neuron(weights=[2.0, 3.0], bias=1.0),
            Neuron(weights=[4.0, 5.0], bias=0.0),
        ]),
        Layer([
            Neuron(weights=[2.0, 3.0], bias=1.0),
        ]),
    ])

    predictions = network.forward(inputs)

    loss_gradients = loss_function.backward(
        predictions=predictions,
        targets=targets,
    )

    input_gradients = network.backward(
        output_gradients=loss_gradients,
    )

    def loss_for_input(value: float) -> float:
        test_inputs = [inputs[0], value]

        predictions = network.forward(test_inputs)

        return loss_function.forward(
            predictions=predictions,
            targets=targets,
        )

    checker = GradientChecker(
        tolerance=1e-5,
    )

    numerical_gradient = checker.numerical_gradient(
        parameter=inputs[1],
        loss_function=loss_for_input,
    )

    assert checker.check(
        analytical_gradient=input_gradients[1],
        numerical_gradient=numerical_gradient,
    )