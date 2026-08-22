from llm_from_scratch.core.feed_forward import FeedForward

def _feed_forward_loss(
    feed_forward: FeedForward,
    inputs: list[list[float]],
) -> float:
    """Calculate a simple scalar loss for gradient checking."""

    output = feed_forward.forward(inputs)

    return sum(
        value
        for row in output
        for value in row
    )

def test_feed_forward_output_has_correct_shape():
    feed_forward = FeedForward(
        input_size=3,
        hidden_size=4,
        output_size=2,
    )

    inputs = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ]

    output = feed_forward.forward(inputs)

    assert len(output) == 2
    assert len(output[0]) == 2
    assert len(output[1]) == 2


def test_feed_forward_relu_removes_negative_values():
    feed_forward = FeedForward(
        input_size=2,
        hidden_size=2,
        output_size=2,
    )

    feed_forward.input_weights = [
        [1.0, 0.0],
        [0.0, 1.0],
    ]

    feed_forward.input_biases = [
        -2.0,
        -2.0,
    ]

    output = feed_forward.forward(
        [
            [1.0, 3.0],
        ]
    )

    assert feed_forward.hidden == [
        [0.0, 1.0],
    ]

    assert output == [
        [0.1, 0.1],
    ]


def test_feed_forward_uses_second_projection():
    feed_forward = FeedForward(
        input_size=2,
        hidden_size=2,
        output_size=1,
    )

    feed_forward.input_weights = [
        [1.0, 0.0],
        [0.0, 1.0],
    ]

    feed_forward.output_weights = [
        [2.0],
        [3.0],
    ]

    output = feed_forward.forward(
        [
            [1.0, 2.0],
        ]
    )

    assert output == [
        [8.0],
    ]


def test_feed_forward_backward_returns_correct_shape():
    feed_forward = FeedForward(
        input_size=3,
        hidden_size=4,
        output_size=2,
    )

    inputs = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ]

    feed_forward.forward(inputs)

    output_gradients = [
        [0.1, 0.2],
        [0.3, 0.4],
    ]

    input_gradients = feed_forward.backward(
        output_gradients,
    )

    assert len(input_gradients) == 2
    assert len(input_gradients[0]) == 3
    assert len(input_gradients[1]) == 3

def test_feed_forward_backward_calculates_output_weight_gradients():
    feed_forward = FeedForward(
        input_size=2,
        hidden_size=2,
        output_size=1,
    )

    feed_forward.input_weights = [
        [1.0, 0.0],
        [0.0, 1.0],
    ]

    feed_forward.output_weights = [
        [2.0],
        [3.0],
    ]

    inputs = [
        [1.0, 2.0],
    ]

    feed_forward.forward(inputs)

    feed_forward.backward(
        [[1.0]],
    )

    assert feed_forward.output_weight_gradients == [
        [1.0],
        [2.0],
    ]

def test_feed_forward_backward_relu_blocks_negative_gradient():
    feed_forward = FeedForward(
        input_size=2,
        hidden_size=2,
        output_size=1,
    )

    feed_forward.input_weights = [
        [1.0, 0.0],
        [0.0, 1.0],
    ]

    feed_forward.output_weights = [
        [1.0],
        [1.0],
    ]

    feed_forward.forward(
        [
            [-1.0, 2.0],
        ]
    )

    input_gradients = feed_forward.backward(
        [[1.0]],
    )

    assert input_gradients == [
        [0.0, 1.0],
    ]


def test_feed_forward_input_gradients_match_numerical_gradient():
    feed_forward = FeedForward(
        input_size=2,
        hidden_size=2,
        output_size=2,
    )

    feed_forward.input_weights = [
        [0.2, -0.3],
        [0.4, 0.5],
    ]

    feed_forward.input_biases = [
        0.1,
        -0.2,
    ]

    feed_forward.output_weights = [
        [0.3, -0.4],
        [0.5, 0.6],
    ]

    feed_forward.output_biases = [
        0.1,
        -0.1,
    ]

    inputs = [
        [0.7, -0.2],
    ]

    output = feed_forward.forward(inputs)

    feed_forward.backward(
        [
            [1.0, 1.0],
        ],
    )

    epsilon = 1e-5

    for row in range(len(inputs)):
        for column in range(len(inputs[row])):
            original = inputs[row][column]

            inputs[row][column] = original + epsilon

            positive_loss = _feed_forward_loss(
                feed_forward,
                inputs,
            )

            inputs[row][column] = original - epsilon

            negative_loss = _feed_forward_loss(
                feed_forward,
                inputs,
            )

            inputs[row][column] = original

            numerical_gradient = (
                positive_loss - negative_loss
            ) / (2 * epsilon)

            analytical_gradient = (
                feed_forward.input_gradients[row][column]
            )

            assert abs(
                numerical_gradient - analytical_gradient
            ) < 1e-5


def test_feed_forward_weight_gradients_match_numerical_gradient():
    feed_forward = FeedForward(
        input_size=2,
        hidden_size=2,
        output_size=2,
    )

    feed_forward.input_weights = [
        [0.2, -0.3],
        [0.4, 0.5],
    ]

    feed_forward.input_biases = [
        0.1,
        -0.2,
    ]

    feed_forward.output_weights = [
        [0.3, -0.4],
        [0.5, 0.6],
    ]

    feed_forward.output_biases = [
        0.1,
        -0.1,
    ]

    inputs = [
        [0.7, -0.2],
    ]

    feed_forward.forward(inputs)

    feed_forward.backward(
        [
            [1.0, 1.0],
        ],
    )

    epsilon = 1e-5

    for row in range(len(feed_forward.input_weights)):
        for column in range(
            len(feed_forward.input_weights[row])
        ):
            original = feed_forward.input_weights[row][column]

            feed_forward.input_weights[row][column] = (
                original + epsilon
            )

            positive_loss = _feed_forward_loss(
                feed_forward,
                inputs,
            )

            feed_forward.input_weights[row][column] = (
                original - epsilon
            )

            negative_loss = _feed_forward_loss(
                feed_forward,
                inputs,
            )

            feed_forward.input_weights[row][column] = original

            numerical_gradient = (
                positive_loss - negative_loss
            ) / (2 * epsilon)

            analytical_gradient = (
                feed_forward.input_weight_gradients[row][column]
            )

            assert abs(
                numerical_gradient - analytical_gradient
            ) < 1e-5