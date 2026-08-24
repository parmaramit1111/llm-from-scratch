"""
Concept: Character Dataset

A character dataset converts text into next-character
training examples.

Example:

    text = "hello"

Vocabulary:

    h → 0
    e → 1
    l → 2
    o → 3

Training examples:

    0 → 1
    1 → 2
    2 → 2
    2 → 3

Each example contains:

    input_values
    targets
"""

from llm_from_scratch.language.vocabulary import Vocabulary


class CharacterDataset:
    """Create next-character training examples from text."""

    def __init__(
        self,
        text: str,
        vocabulary: Vocabulary,
    ):
        self.text = text
        self.vocabulary = vocabulary

    def __len__(self) -> int:
        """Return the number of training examples."""
        return max(0, len(self.text) - 1)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[list[float], list[float]]:
        """Return one input-target training example."""

        input_token = self.text[index]
        target_token = self.text[index + 1]

        input_id = self.vocabulary.encode(input_token)
        target_id = self.vocabulary.encode(target_token)

        return (
            [float(input_id)],
            [float(target_id)],
        )