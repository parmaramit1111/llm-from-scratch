from llm_from_scratch.core.residual import Residual


def test_residual_output_has_correct_shape():
    residual = Residual()

    output = residual.forward(
        inputs=[
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        sublayer_output=[
            [0.5, 0.5],
            [1.0, 1.0],
        ],
    )

    assert len(output) == 2
    assert len(output[0]) == 2
    assert len(output[1]) == 2


def test_residual_adds_input_and_sublayer_output():
    residual = Residual()

    output = residual.forward(
        inputs=[
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        sublayer_output=[
            [0.5, 0.5],
            [1.0, 1.0],
        ],
    )

    assert output == [
        [1.5, 2.5],
        [4.0, 5.0],
    ]


def test_residual_backward_returns_same_gradient_for_both_paths():
    residual = Residual()

    input_gradients, sublayer_gradients = residual.backward(
        [
            [0.1, 0.2],
            [0.3, 0.4],
        ]
    )

    expected = [
        [0.1, 0.2],
        [0.3, 0.4],
    ]

    assert input_gradients == expected
    assert sublayer_gradients == expected