from llm_from_scratch.language.dataset import CharacterDataset
from llm_from_scratch.language.vocabulary import Vocabulary


def test_character_dataset_length():
    vocabulary = Vocabulary(["h", "e", "l", "o"])

    dataset = CharacterDataset(
        text="hello",
        vocabulary=vocabulary,
    )

    assert len(dataset) == 4


def test_character_dataset_creates_next_character_pairs():
    vocabulary = Vocabulary(["h", "e", "l", "o"])

    dataset = CharacterDataset(
        text="hello",
        vocabulary=vocabulary,
    )

    assert dataset[0] == ([0.0], [1.0])
    assert dataset[1] == ([1.0], [2.0])
    assert dataset[2] == ([2.0], [2.0])
    assert dataset[3] == ([2.0], [3.0])