import pytest

from llm_from_scratch.core.layer_normalization import (
    LayerNormalization,
)

def _layer_normalization_loss(
    layer_norm: LayerNormalization,
    inputs: list[list[float]],
) -> float:
    """Calculate a simple scalar loss for gradient checking."""

    output = layer_norm.forward(inputs)

    return sum(
        value
        for row in output
        for value in row
    )

def test_layer_normalization_output_has_correct_shape():
    layer_norm = LayerNormalization(
        normalized_size=3,
    )

    output = layer_norm.forward(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )

    assert len(output) == 2
    assert len(output[0]) == 3
    assert len(output[1]) == 3


def test_layer_normalization_rows_have_zero_mean():
    layer_norm = LayerNormalization(
        normalized_size=3,
    )

    output = layer_norm.forward(
        [
            [1.0, 2.0, 3.0],
        ]
    )

    mean = sum(output[0]) / len(output[0])

    assert abs(mean) < 1e-5


def test_layer_normalization_rows_have_unit_variance():
    layer_norm = LayerNormalization(
        normalized_size=3,
    )

    output = layer_norm.forward(
        [
            [1.0, 2.0, 3.0],
        ]
    )

    mean = sum(output[0]) / len(output[0])

    variance = sum(
        (value - mean) ** 2
        for value in output[0]
    ) / len(output[0])

    assert abs(variance - 1.0) < 1e-4


def test_layer_normalization_uses_gamma_and_beta():
    layer_norm = LayerNormalization(
        normalized_size=3,
    )

    layer_norm.gamma = [
        2.0,
        3.0,
        4.0,
    ]

    layer_norm.beta = [
        1.0,
        2.0,
        3.0,
    ]

    output = layer_norm.forward(
        [
            [1.0, 2.0, 3.0],
        ]
    )

    normalized = layer_norm.normalized[0]

    expected = [
        2.0 * normalized[0] + 1.0,
        3.0 * normalized[1] + 2.0,
        4.0 * normalized[2] + 3.0,
    ]

    assert output == [expected]

def test_layer_normalization_backward_returns_correct_shape():
    layer_norm = LayerNormalization(
        normalized_size=3,
    )

    layer_norm.forward(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )

    input_gradients = layer_norm.backward(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ]
    )

    assert len(input_gradients) == 2
    assert len(input_gradients[0]) == 3
    assert len(input_gradients[1]) == 3


def test_layer_normalization_backward_calculates_gamma_gradients():
    layer_norm = LayerNormalization(
        normalized_size=3,
    )

    layer_norm.forward(
        [
            [1.0, 2.0, 3.0],
        ]
    )

    layer_norm.backward(
        [
            [0.1, 0.2, 0.3],
        ]
    )

    normalized = layer_norm.normalized[0]

    expected = [
        0.1 * normalized[0],
        0.2 * normalized[1],
        0.3 * normalized[2],
    ]

    assert layer_norm.gamma_gradients == pytest.approx(
        expected
    )


def test_layer_normalization_backward_calculates_beta_gradients():
    layer_norm = LayerNormalization(
        normalized_size=3,
    )

    layer_norm.forward(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )

    layer_norm.backward(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ]
    )

    assert layer_norm.beta_gradients == pytest.approx([
        0.5,
        0.7,
        0.9,
    ])

def test_layer_normalization_input_gradients_match_numerical_gradient():
    layer_norm = LayerNormalization(
        normalized_size=3,
    )

    layer_norm.gamma = [
        1.2,
        0.8,
        1.5,
    ]

    layer_norm.beta = [
        0.1,
        -0.2,
        0.3,
    ]

    inputs = [
        [0.7, -0.4, 1.1],
    ]

    layer_norm.forward(inputs)

    layer_norm.backward(
        [
            [1.0, 1.0, 1.0],
        ],
    )

    epsilon = 1e-5

    for row in range(len(inputs)):
        for column in range(len(inputs[row])):
            original = inputs[row][column]

            inputs[row][column] = original + epsilon

            positive_loss = _layer_normalization_loss(
                layer_norm,
                inputs,
            )

            inputs[row][column] = original - epsilon

            negative_loss = _layer_normalization_loss(
                layer_norm,
                inputs,
            )

            inputs[row][column] = original

            numerical_gradient = (
                positive_loss - negative_loss
            ) / (2 * epsilon)

            analytical_gradient = (
                layer_norm.input_gradients[row][column]
            )

            assert numerical_gradient == pytest.approx(
                analytical_gradient,
                abs=1e-5,
            )

def test_layer_normalization_gamma_gradients_match_numerical_gradient():
    layer_norm = LayerNormalization(
        normalized_size=3,
    )

    layer_norm.gamma = [
        1.2,
        0.8,
        1.5,
    ]

    layer_norm.beta = [
        0.1,
        -0.2,
        0.3,
    ]

    inputs = [
        [0.7, -0.4, 1.1],
    ]

    layer_norm.forward(inputs)

    layer_norm.backward(
        [
            [1.0, 1.0, 1.0],
        ],
    )

    epsilon = 1e-5

    for column in range(len(layer_norm.gamma)):
        original = layer_norm.gamma[column]

        layer_norm.gamma[column] = (
            original + epsilon
        )

        positive_loss = _layer_normalization_loss(
            layer_norm,
            inputs,
        )

        layer_norm.gamma[column] = (
            original - epsilon
        )

        negative_loss = _layer_normalization_loss(
            layer_norm,
            inputs,
        )

        layer_norm.gamma[column] = original

        numerical_gradient = (
            positive_loss - negative_loss
        ) / (2 * epsilon)

        analytical_gradient = (
            layer_norm.gamma_gradients[column]
        )

        assert numerical_gradient == pytest.approx(
            analytical_gradient,
            abs=1e-5,
        )

def test_layer_normalization_beta_gradients_match_numerical_gradient():
    layer_norm = LayerNormalization(
        normalized_size=3,
    )

    layer_norm.gamma = [
        1.2,
        0.8,
        1.5,
    ]

    layer_norm.beta = [
        0.1,
        -0.2,
        0.3,
    ]

    inputs = [
        [0.7, -0.4, 1.1],
    ]

    layer_norm.forward(inputs)

    layer_norm.backward(
        [
            [1.0, 1.0, 1.0],
        ],
    )

    epsilon = 1e-5

    for column in range(len(layer_norm.beta)):
        original = layer_norm.beta[column]

        layer_norm.beta[column] = (
            original + epsilon
        )

        positive_loss = _layer_normalization_loss(
            layer_norm,
            inputs,
        )

        layer_norm.beta[column] = (
            original - epsilon
        )

        negative_loss = _layer_normalization_loss(
            layer_norm,
            inputs,
        )

        layer_norm.beta[column] = original

        numerical_gradient = (
            positive_loss - negative_loss
        ) / (2 * epsilon)

        analytical_gradient = (
            layer_norm.beta_gradients[column]
        )

        assert numerical_gradient == pytest.approx(
            analytical_gradient,
            abs=1e-5,
        )