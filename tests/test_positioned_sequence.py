from llm_from_scratch.core.embedding import Embedding
from llm_from_scratch.core.positional_embedding import (
    PositionalEmbedding,
)
from llm_from_scratch.language.positioned_sequence import (
    PositionedSequence,
)


def test_positioned_sequence_forward_combines_embeddings():
    embedding = Embedding(
        vocabulary_size=4,
        embedding_size=3,
    )

    positional_embedding = PositionalEmbedding(
        maximum_sequence_length=4,
        embedding_size=3,
    )

    embedding.weights = [
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
        [3.0, 3.0, 3.0],
        [4.0, 4.0, 4.0],
    ]

    positional_embedding.weights = [
        [0.1, 0.1, 0.1],
        [0.2, 0.2, 0.2],
        [0.3, 0.3, 0.3],
        [0.4, 0.4, 0.4],
    ]

    sequence = PositionedSequence(
        embedding=embedding,
        positional_embedding=positional_embedding,
    )

    output = sequence.forward(
        token_ids=[1, 2],
    )

    assert output == [
        [2.1, 2.1, 2.1],
        [3.2, 3.2, 3.2],
    ]


def test_positioned_sequence_forward_preserves_sequence_shape():
    embedding = Embedding(
        vocabulary_size=4,
        embedding_size=3,
    )

    positional_embedding = PositionalEmbedding(
        maximum_sequence_length=4,
        embedding_size=3,
    )

    sequence = PositionedSequence(
        embedding=embedding,
        positional_embedding=positional_embedding,
    )

    output = sequence.forward(
        token_ids=[0, 1, 2],
    )

    assert len(output) == 3
    assert len(output[0]) == 3
    assert len(output[1]) == 3
    assert len(output[2]) == 3


def test_positioned_sequence_backward_propagates_gradients():
    embedding = Embedding(
        vocabulary_size=4,
        embedding_size=3,
    )

    positional_embedding = PositionalEmbedding(
        maximum_sequence_length=4,
        embedding_size=3,
    )

    sequence = PositionedSequence(
        embedding=embedding,
        positional_embedding=positional_embedding,
    )

    sequence.forward(
        token_ids=[1, 2],
    )

    output_gradients = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]

    input_gradients = sequence.backward(
        output_gradients,
    )

    assert input_gradients == output_gradients

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