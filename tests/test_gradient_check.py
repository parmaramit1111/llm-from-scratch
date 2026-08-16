from typing import Callable

from llm_from_scratch.core.gradient_check import GradientChecker
from llm_from_scratch.core.layer import Layer
from llm_from_scratch.core.loss import MeanSquaredError
from llm_from_scratch.core.neuron import Neuron


def test_numerical_gradient():
    """A numerical gradient should approximate the analytical derivative."""

    checker = GradientChecker()

    parameter = 3.0

    numerical_gradient = checker.numerical_gradient(
        parameter=parameter,
        loss_function=lambda value: value**2,
    )

    assert abs(numerical_gradient - 6.0) < 1e-5

def test_gradient_check_passes_for_matching_gradients():
    """Gradient check should pass when gradients are within tolerance."""

    checker = GradientChecker(
        tolerance=1e-5,
    )

    result = checker.check(
        analytical_gradient=6.0,
        numerical_gradient=6.000001,
    )

    assert result is True

def test_gradient_check_fails_for_different_gradients():
    """Gradient check should fail when gradients exceed tolerance."""

    checker = GradientChecker(
        tolerance=1e-5,
    )

    result = checker.check(
        analytical_gradient=6.0,
        numerical_gradient=6.1,
    )

    assert result is False

def test_neuron_weight_gradient():
    inputs = [3.0]
    targets = [9.0]

    neuron = Neuron(
        weights=[2.0],
        bias=1.0,
    )

    loss_function = MeanSquaredError()

    # 1. Forward pass
    prediction = neuron.forward(inputs)

    # 2. Calculate loss gradient
    loss_gradient = loss_function.backward(
        predictions=[prediction],
        targets=targets,
    )

    # 3. Analytical gradient from Neuron.backward()
    neuron.backward(
        output_gradient=loss_gradient[0],
    )

    analytical_gradient = neuron.gradient[0]

    # 4. Function used by GradientChecker.
    #    It calculates loss for a given weight.
    def loss_for_weight(value: float) -> float:
        neuron.weights[0] = value

        prediction = neuron.forward(inputs)

        return loss_function.forward(
            predictions=[prediction],
            targets=targets,
        )

    # 5. Calculate numerical gradient
    checker = GradientChecker(
        tolerance=1e-5,
    )

    numerical_gradient = checker.numerical_gradient(
        parameter=2.0,
        loss_function=loss_for_weight,
    )

    # 6. Compare analytical vs numerical gradient
    result = checker.check(
        analytical_gradient=analytical_gradient,
        numerical_gradient=numerical_gradient,
    )

    assert result is True

def test_neuron_second_weight_gradient():
    """A neuron should calculate the correct gradient for its second weight."""

    inputs = [3.0, 4.0]
    targets = [9.0]

    neuron = Neuron(
        weights=[2.0, 3.0],
        bias=1.0,
    )

    loss_function = MeanSquaredError()

    # 1. Forward pass
    prediction = neuron.forward(inputs)

    # 2. Calculate loss gradient
    loss_gradient = loss_function.backward(
        predictions=[prediction],
        targets=targets,
    )

    # 3. Analytical gradient from Neuron.backward()
    neuron.backward(
        output_gradient=loss_gradient[0],
    )

    analytical_gradient = neuron.gradient[1]

    # 4. Function used by GradientChecker.
    #    It calculates loss for a given second weight.
    def loss_for_weight(value: float) -> float:
        neuron.weights[1] = value

        prediction = neuron.forward(inputs)

        return loss_function.forward(
            predictions=[prediction],
            targets=targets,
        )

    # 5. Calculate numerical gradient
    checker = GradientChecker(
        tolerance=1e-5,
    )

    numerical_gradient = checker.numerical_gradient(
        parameter=3.0,
        loss_function=loss_for_weight,
    )

    # 6. Compare analytical vs numerical gradient
    result = checker.check(
        analytical_gradient=analytical_gradient,
        numerical_gradient=numerical_gradient,
    )

    assert result is True

def test_neuron_bias_gradient():
    inputs = [3.0, 4.0]
    targets = [9.0]

    neuron = Neuron(
        weights=[2.0, 3.0],
        bias=1.0,
    )

    loss_function = MeanSquaredError()

    # 1. Forward pass
    prediction = neuron.forward(inputs)

    # 2. Calculate loss gradient
    loss_gradient = loss_function.backward(
        predictions=[prediction],
        targets=targets,
    )

    # 3. Analytical gradient from Neuron.backward()
    neuron.backward(
        output_gradient=loss_gradient[0],
    )

    analytical_gradient = neuron.bias_gradient

    # 4. Function used by GradientChecker.
    #    It calculates loss for a given bias.
    def loss_for_bias(value: float) -> float:
        neuron.bias = value

        prediction = neuron.forward(inputs)

        return loss_function.forward(
            predictions=[prediction],
            targets=targets,
        )

    # 5. Calculate numerical gradient
    checker = GradientChecker(
        tolerance=1e-5,
    )

    numerical_gradient = checker.numerical_gradient(
        parameter=1.0,
        loss_function=loss_for_bias,
    )

    # 6. Compare analytical vs numerical gradient
    result = checker.check(
        analytical_gradient=analytical_gradient,
        numerical_gradient=numerical_gradient,
    )

    assert result is True


def test_layer_input_gradients():
    """A layer should correctly aggregate input gradients from all neurons."""

    loss_function = MeanSquaredError()

    layer = Layer([
        Neuron(weights=[2.0, 3.0], bias=1.0),
        Neuron(weights=[4.0, 5.0], bias=0.0),
    ])

    inputs = [2.0, 3.0]
    targets = [6.0, 9.0]

    predictions = layer.forward(inputs)

    loss_gradients = loss_function.backward(
        predictions=predictions,
        targets=targets,
    )

    input_gradients = layer.backward(
        output_gradients=loss_gradients,
    )

    assert input_gradients == [72.0, 94.0]

def test_layer_input_gradients_with_gradient_checker():
    """A layer input gradient should match the numerical gradient."""
    loss_function = MeanSquaredError()

    layer = Layer([
        Neuron(weights=[2.0, 3.0], bias=1.0),
        Neuron(weights=[4.0, 5.0], bias=0.0),
    ])

    inputs = [2.0, 3.0]
    targets = [6.0, 9.0]

    predictions = layer.forward(inputs)

    loss_gradients = loss_function.backward(
        predictions=predictions,
        targets=targets,
    )

    input_gradients = layer.backward(
        output_gradients=loss_gradients,
    )

    def loss_for_input(value: float) -> float:
        test_inputs = [value, inputs[1]]

        predictions = layer.forward(test_inputs)

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

    analytical_gradient = input_gradients[0]

    assert checker.check(
        analytical_gradient=analytical_gradient,
        numerical_gradient=numerical_gradient,
    )