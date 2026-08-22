from llm_from_scratch.core.embedding import Embedding
from llm_from_scratch.language.embedded_sequence import EmbeddedSequence


def test_embedded_sequence_forward_flattens_vectors():
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

    sequence = EmbeddedSequence(
        embedding=embedding,
    )

    result = sequence.forward(
        token_ids=[1, 2],
    )

    assert result == [
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
    ]


def test_embedded_sequence_backward_reshapes_gradients():
    embedding = Embedding(
        vocabulary_size=4,
        embedding_size=3,
    )

    sequence = EmbeddedSequence(
        embedding=embedding,
    )

    sequence.backward(
        token_ids=[1, 2],
        output_gradients=[
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
        ],
    )

    assert embedding.gradients[1] == [
        0.1,
        0.2,
        0.3,
    ]

    assert embedding.gradients[2] == [
        0.4,
        0.5,
        0.6,
    ]