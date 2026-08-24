import pytest

from llm_from_scratch.core.embedding import Embedding
from llm_from_scratch.core.gradient_check import GradientChecker

def test_embedding_forward_returns_vectors():
    embedding = Embedding(
        vocabulary_size=4,
        embedding_size=3,
    )

    vectors = embedding.forward(
        token_ids=[1, 2],
    )

    assert len(vectors) == 2
    assert len(vectors[0]) == 3
    assert len(vectors[1]) == 3


def test_embedding_forward_returns_correct_rows():
    embedding = Embedding(
        vocabulary_size=4,
        embedding_size=3,
    )

    embedding.weights = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
        [1.0, 1.1, 1.2],
    ]

    vectors = embedding.forward(
        token_ids=[1, 2],
    )

    assert vectors == [
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
    ]


def test_embedding_backward_accumulates_gradients():
    embedding = Embedding(
        vocabulary_size=4,
        embedding_size=3,
    )

    embedding.backward(
        token_ids=[1, 2],
        output_gradients=[
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ],
    )

    assert embedding.gradients[0] == [0.0, 0.0, 0.0]
    assert embedding.gradients[1] == [0.1, 0.2, 0.3]
    assert embedding.gradients[2] == [0.4, 0.5, 0.6]
    assert embedding.gradients[3] == [0.0, 0.0, 0.0]


def test_embedding_backward_accumulates_repeated_tokens():
    embedding = Embedding(
        vocabulary_size=4,
        embedding_size=3,
    )

    embedding.backward(
        token_ids=[2, 2],
        output_gradients=[
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ],
    )

    assert embedding.gradients[2] == pytest.approx([
        0.5,
        0.7,
        0.9,
    ])


def test_embedding_weight_gradient():
    """Embedding weight gradients should match numerical gradients."""

    embedding = Embedding(
        vocabulary_size=3,
        embedding_size=2,
    )

    # Use deterministic values so the test is reproducible.
    embedding.weights = [
        [0.1, 0.2],
        [0.3, 0.4],
        [0.5, 0.6],
    ]

    token_ids = [1]

    def loss_for_weight(value: float) -> float:
        embedding.weights[1][0] = value

        vector = embedding.forward(token_ids)[0]

        # Simple scalar loss:
        # L = sum(vector)
        return sum(vector)

    # Analytical gradient.
    embedding.backward(
        token_ids=token_ids,
        output_gradients=[
            [1.0, 1.0],
        ],
    )

    analytical_gradient = embedding.gradients[1][0]

    checker = GradientChecker(
        tolerance=1e-5,
    )

    numerical_gradient = checker.numerical_gradient(
        parameter=0.3,
        loss_function=loss_for_weight,
    )

    assert checker.check(
        analytical_gradient=analytical_gradient,
        numerical_gradient=numerical_gradient,
    )