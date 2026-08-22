import pytest

from llm_from_scratch.core.transformer_block import (
    TransformerBlock,
)


def _create_transformer() -> TransformerBlock:
    return TransformerBlock(
        input_size=3,
        attention_size=3,
        feed_forward_hidden_size=6,
    )


def _create_inputs() -> list[list[float]]:
    return [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]


def test_transformer_block_output_has_correct_shape():
    transformer = _create_transformer()

    output = transformer.forward(
        _create_inputs(),
    )

    assert len(output) == 2
    assert len(output[0]) == 3
    assert len(output[1]) == 3


def test_transformer_block_stores_intermediate_outputs():
    transformer = _create_transformer()

    inputs = _create_inputs()

    output = transformer.forward(inputs)

    assert transformer.inputs == inputs

    assert len(transformer.attention_output) == 2
    assert len(transformer.attention_residual_output) == 2
    assert len(transformer.attention_normalized_output) == 2

    assert len(transformer.feed_forward_output) == 2
    assert len(transformer.feed_forward_residual_output) == 2

    assert output == transformer.outputs


def test_transformer_block_layer_normalization():
    transformer = _create_transformer()

    output = transformer.forward(
        _create_inputs(),
    )

    for row in output:
        mean = sum(row) / len(row)

        variance = sum(
            (value - mean) ** 2
            for value in row
        ) / len(row)

        assert abs(mean) < 1e-5
        assert abs(variance - 1.0) < 2e-3


def test_transformer_block_backward_returns_correct_shape():
    transformer = _create_transformer()

    transformer.forward(
        _create_inputs(),
    )

    input_gradients = transformer.backward(
        [
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
    )

    assert len(input_gradients) == 2
    assert len(input_gradients[0]) == 3
    assert len(input_gradients[1]) == 3


def test_transformer_block_backward_stores_gradient_paths():
    transformer = _create_transformer()

    transformer.forward(
        _create_inputs(),
    )

    output_gradients = [
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ]

    transformer.backward(
        output_gradients,
    )

    for input_row, sublayer_row in zip(
        transformer.residual_input_gradients,
        transformer.residual_sublayer_gradients,
        strict=True,
    ):
        assert input_row == pytest.approx(
            sublayer_row,
        )


def test_transformer_block_feed_forward_output_has_correct_shape():
    transformer = _create_transformer()

    output = transformer.forward(
        _create_inputs(),
    )

    assert len(output) == 2
    assert len(output[0]) == 3
    assert len(output[1]) == 3


def test_transformer_block_runs_feed_forward_stage():
    transformer = _create_transformer()

    inputs = _create_inputs()

    output = transformer.forward(inputs)

    assert len(transformer.attention_normalized_output) == 2
    assert len(transformer.feed_forward_output) == 2
    assert len(transformer.feed_forward_residual_output) == 2

    assert output == transformer.outputs