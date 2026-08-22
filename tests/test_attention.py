import math

from llm_from_scratch.core.attention import SelfAttention


def _attention_loss(attention, inputs):
    output = attention.forward(inputs)

    return sum(
        value
        for row in output
        for value in row
    )

def test_attention_output_has_correct_shape():
    attention = SelfAttention(
        input_size=3,
        attention_size=2,
    )

    inputs = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]

    output = attention.forward(inputs)

    assert len(output) == 2
    assert len(output[0]) == 2
    assert len(output[1]) == 2


def test_attention_weights_sum_to_one():
    attention = SelfAttention(
        input_size=3,
        attention_size=2,
    )

    inputs = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]

    attention.forward(inputs)

    for row in attention.attention_weights:
        assert abs(sum(row) - 1.0) < 1e-9


def test_attention_weights_are_non_negative():
    attention = SelfAttention(
        input_size=3,
        attention_size=2,
    )

    inputs = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]

    attention.forward(inputs)

    for row in attention.attention_weights:
        for value in row:
            assert value >= 0.0



def test_attention_calculation_with_known_weights():
    attention = SelfAttention(
        input_size=2,
        attention_size=2,
    )

    attention.query_weights = [
        [1.0, 0.0],
        [0.0, 1.0],
    ]

    attention.key_weights = [
        [1.0, 0.0],
        [0.0, 1.0],
    ]

    attention.value_weights = [
        [1.0, 0.0],
        [0.0, 1.0],
    ]

    inputs = [
        [1.0, 0.0],
        [0.0, 1.0],
    ]

    output = attention.forward(inputs)

    scale = math.sqrt(2.0)

    expected_first_weight = (
        math.exp(1.0 / scale)
        / (
            math.exp(1.0 / scale)
            + math.exp(0.0)
        )
    )

    expected_second_weight = (
        math.exp(0.0)
        / (
            math.exp(1.0 / scale)
            + math.exp(0.0)
        )
    )

    assert abs(
        attention.attention_weights[0][0]
        - expected_first_weight
    ) < 1e-9

    assert abs(
        attention.attention_weights[0][1]
        - expected_second_weight
    ) < 1e-9

    assert abs(
        attention.attention_weights[1][0]
        - expected_second_weight
    ) < 1e-9

    assert abs(
        attention.attention_weights[1][1]
        - expected_first_weight
    ) < 1e-9

    assert abs(
        output[0][0]
        - expected_first_weight
    ) < 1e-9

    assert abs(
        output[0][1]
        - expected_second_weight
    ) < 1e-9

    assert abs(
        output[1][0]
        - expected_second_weight
    ) < 1e-9

    assert abs(
        output[1][1]
        - expected_first_weight
    ) < 1e-9


def test_attention_backward_calculates_value_gradients():
    attention = SelfAttention(
        input_size=2,
        attention_size=2,
    )

    inputs = [
        [1.0, 0.0],
        [0.0, 1.0],
    ]

    attention.forward(inputs)

    output_gradients = [
        [1.0, 0.0],
        [0.0, 1.0],
    ]

    attention.backward(output_gradients)

    assert len(attention.attention_weight_gradients) == 2
    assert len(attention.value_gradients) == 2

    assert len(attention.attention_weight_gradients[0]) == 2
    assert len(attention.value_gradients[0]) == 2


def test_attention_backward_propagates_through_scaling():
    attention = SelfAttention(
        input_size=2,
        attention_size=2,
    )

    inputs = [
        [1.0, 0.0],
        [0.0, 1.0],
    ]

    attention.forward(inputs)

    output_gradients = [
        [1.0, 0.0],
        [0.0, 1.0],
    ]

    attention.backward(output_gradients)

    scale = math.sqrt(2.0)

    for scaled_row, score_row in zip(
        attention.scaled_score_gradients,
        attention.score_gradients,
        strict=True,
    ):
        for scaled_gradient, score_gradient in zip(
            scaled_row,
            score_row,
            strict=True,
        ):
            assert abs(
                score_gradient
                - scaled_gradient / scale
            ) < 1e-9


def test_attention_backward_calculates_weight_gradients():
    attention = SelfAttention(
        input_size=2,
        attention_size=2,
    )

    inputs = [
        [1.0, 0.0],
        [0.0, 1.0],
    ]

    attention.forward(inputs)

    output_gradients = [
        [1.0, 0.0],
        [0.0, 1.0],
    ]

    attention.backward(output_gradients)

    assert len(attention.query_weight_gradients) == 2
    assert len(attention.key_weight_gradients) == 2
    assert len(attention.value_weight_gradients) == 2

    assert len(attention.query_weight_gradients[0]) == 2
    assert len(attention.key_weight_gradients[0]) == 2
    assert len(attention.value_weight_gradients[0]) == 2


def test_attention_backward_returns_input_gradients():
    attention = SelfAttention(
        input_size=2,
        attention_size=2,
    )

    inputs = [
        [1.0, 0.0],
        [0.0, 1.0],
    ]

    attention.forward(inputs)

    output_gradients = [
        [1.0, 0.0],
        [0.0, 1.0],
    ]

    input_gradients = attention.backward(
        output_gradients,
    )

    assert len(input_gradients) == 2
    assert len(input_gradients[0]) == 2
    assert len(input_gradients[1]) == 2

def test_attention_query_weight_gradient_matches_numerical_gradient():
    attention = SelfAttention(
        input_size=2,
        attention_size=2,
    )

    inputs = [
        [0.5, -0.2],
        [0.3, 0.8],
    ]

    attention.forward(inputs)

    output_gradients = [
        [1.0, 1.0],
        [1.0, 1.0],
    ]

    input_gradients = attention.backward(
        output_gradients,
    )

    epsilon = 1e-6

    row = 0
    column = 0

    original = attention.query_weights[row][column]

    attention.query_weights[row][column] = (
        original + epsilon
    )

    positive_loss = _attention_loss(
        attention,
        inputs,
    )

    attention.query_weights[row][column] = (
        original - epsilon
    )

    negative_loss = _attention_loss(
        attention,
        inputs,
    )

    attention.query_weights[row][column] = original

    numerical_gradient = (
        positive_loss - negative_loss
    ) / (2 * epsilon)

    analytical_gradient = (
        attention.query_weight_gradients[row][column]
    )

    assert abs(
        analytical_gradient - numerical_gradient
    ) < 1e-5

def test_attention_key_weight_gradient_matches_numerical_gradient():
    attention = SelfAttention(
        input_size=2,
        attention_size=2,
    )

    inputs = [
        [0.5, -0.2],
        [0.3, 0.8],
    ]

    attention.forward(inputs)

    output_gradients = [
        [1.0, 1.0],
        [1.0, 1.0],
    ]

    attention.backward(output_gradients)

    epsilon = 1e-6

    row = 0
    column = 0

    original = attention.key_weights[row][column]

    attention.key_weights[row][column] = original + epsilon
    positive_loss = _attention_loss(attention, inputs)

    attention.key_weights[row][column] = original - epsilon
    negative_loss = _attention_loss(attention, inputs)

    attention.key_weights[row][column] = original

    numerical_gradient = (
        positive_loss - negative_loss
    ) / (2 * epsilon)

    analytical_gradient = (
        attention.key_weight_gradients[row][column]
    )

    assert abs(
        analytical_gradient - numerical_gradient
    ) < 1e-5

def test_attention_input_gradient_matches_numerical_gradient():
    attention = SelfAttention(
        input_size=2,
        attention_size=2,
    )

    inputs = [
        [0.5, -0.2],
        [0.3, 0.8],
    ]

    output_gradients = [
        [1.0, 1.0],
        [1.0, 1.0],
    ]

    attention.forward(inputs)

    input_gradients = attention.backward(
        output_gradients,
    )

    epsilon = 1e-6

    row = 0
    column = 0

    original = inputs[row][column]

    inputs[row][column] = original + epsilon
    positive_loss = _attention_loss(attention, inputs)

    inputs[row][column] = original - epsilon
    negative_loss = _attention_loss(attention, inputs)

    inputs[row][column] = original

    numerical_gradient = (
        positive_loss - negative_loss
    ) / (2 * epsilon)

    analytical_gradient = input_gradients[row][column]

    assert abs(
        analytical_gradient - numerical_gradient
    ) < 1e-5