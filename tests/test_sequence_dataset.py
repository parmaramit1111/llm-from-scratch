from llm_from_scratch.language.sequence_dataset import SequenceDataset
from llm_from_scratch.language.vocabulary import Vocabulary


def test_sequence_dataset_length():
    vocabulary = Vocabulary(["h", "e", "l", "o"])

    dataset = SequenceDataset(
        text="hello",
        vocabulary=vocabulary,
        context_size=2,
    )

    assert len(dataset) == 3


def test_sequence_dataset_creates_context_windows():
    vocabulary = Vocabulary(["h", "e", "l", "o"])

    dataset = SequenceDataset(
        text="hello",
        vocabulary=vocabulary,
        context_size=2,
    )

    assert dataset[0] == ([0.0, 1.0], [2.0])
    assert dataset[1] == ([1.0, 2.0], [2.0])
    assert dataset[2] == ([2.0, 2.0], [3.0])