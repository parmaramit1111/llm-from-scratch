import pytest

from llm_from_scratch.models.small_language_model import (
    SmallLanguageModel,
)


def _create_model() -> SmallLanguageModel:
    return SmallLanguageModel(
        vocabulary_size=4,
        embedding_size=3,
        attention_size=3,
        feed_forward_hidden_size=6,
        sequence_length=2,
    )


def test_small_language_model_forward_has_correct_shape():
    model = _create_model()

    probabilities = model.forward(
        [0, 1],
    )

    assert len(probabilities) == 2
    assert len(probabilities[0]) == 4
    assert len(probabilities[1]) == 4


def test_small_language_model_probabilities_sum_to_one():
    model = _create_model()

    probabilities = model.forward(
        [0, 1],
    )

    for row in probabilities:
        assert sum(row) == pytest.approx(1.0)


def test_small_language_model_loss_is_positive():
    model = _create_model()

    model.forward(
        [0, 1],
    )

    loss = model.loss(
        target_token_id=2,
    )

    assert loss > 0.0
    assert loss < 10.0


def test_small_language_model_backward_has_correct_shape():
    model = _create_model()

    model.forward(
        [0, 1],
    )

    gradients = model.backward(
        target_token_id=2,
    )

    assert len(gradients) == 2
    assert len(gradients[0]) == 3
    assert len(gradients[1]) == 3

    assert len(model.output_weight_gradients) == 3
    assert len(model.output_weight_gradients[0]) == 4

    assert len(model.token_embedding_gradients) == 2
    assert len(model.token_embedding_gradients[0]) == 3

    assert len(model.position_embedding_gradients) == 2
    assert len(model.position_embedding_gradients[0]) == 3


def test_small_language_model_update_changes_parameters():
    model = _create_model()

    model.forward(
        [0, 1],
    )

    model.backward(
        target_token_id=2,
    )

    original_output_biases = (
        model.output_biases.copy()
    )

    model.update(
        learning_rate=0.1,
    )

    assert model.output_biases != original_output_biases