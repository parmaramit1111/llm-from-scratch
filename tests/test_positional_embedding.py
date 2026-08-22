from llm_from_scratch.core.positional_embedding import (
    PositionalEmbedding,
)


def test_positional_embedding_forward_returns_correct_shape():
    positional_embedding = PositionalEmbedding(
        maximum_sequence_length=4,
        embedding_size=3,
    )

    token_embeddings = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]

    output = positional_embedding.forward(
        token_embeddings,
    )

    assert len(output) == 2
    assert len(output[0]) == 3
    assert len(output[1]) == 3


def test_positional_embedding_adds_position_vectors():
    positional_embedding = PositionalEmbedding(
        maximum_sequence_length=4,
        embedding_size=3,
    )

    positional_embedding.weights = [
        [0.1, 0.1, 0.1],
        [0.2, 0.2, 0.2],
        [0.3, 0.3, 0.3],
        [0.4, 0.4, 0.4],
    ]

    token_embeddings = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ]

    output = positional_embedding.forward(
        token_embeddings,
    )

    assert output == [
        [1.1, 2.1, 3.1],
        [4.2, 5.2, 6.2],
    ]


def test_positional_embedding_backward_returns_input_gradients():
    positional_embedding = PositionalEmbedding(
        maximum_sequence_length=4,
        embedding_size=3,
    )

    token_embeddings = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ]

    positional_embedding.forward(
        token_embeddings,
    )

    output_gradients = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]

    input_gradients = positional_embedding.backward(
        output_gradients,
    )

    assert input_gradients == output_gradients


def test_positional_embedding_backward_stores_position_gradients():
    positional_embedding = PositionalEmbedding(
        maximum_sequence_length=4,
        embedding_size=3,
    )

    token_embeddings = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ]

    positional_embedding.forward(
        token_embeddings,
    )

    output_gradients = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]

    positional_embedding.backward(
        output_gradients,
    )

    assert positional_embedding.gradients[0] == [
        0.1,
        0.2,
        0.3,
    ]

    assert positional_embedding.gradients[1] == [
        0.4,
        0.5,
        0.6,
    ]