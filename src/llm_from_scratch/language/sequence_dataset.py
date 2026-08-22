"""
Concept: Sequence Dataset

A sequence dataset creates fixed-size context windows
for next-token prediction.

Example:

    text = "hello"
    context_size = 2

Training examples:

    "he" → "l"
    "el" → "l"
    "ll" → "o"

Using token IDs:

    [0, 1] → [2]
    [1, 2] → [2]
    [2, 2] → [3]
"""

from llm_from_scratch.language.vocabulary import Vocabulary


class SequenceDataset:
    """Create fixed-size context and next-token examples."""

    def __init__(
        self,
        text: str,
        vocabulary: Vocabulary,
        context_size: int = 2,
    ):
        self.text = text
        self.vocabulary = vocabulary
        self.context_size = context_size

    def __len__(self) -> int:
        """Return the number of training examples."""
        return len(self.text) - self.context_size

    def __getitem__(
        self,
        index: int,
    ) -> tuple[list[float], list[float]]:
        """Return one context and its next-token target."""

        context = self.text[
            index:index + self.context_size
        ]

        target = self.text[
            index + self.context_size
        ]

        input_ids = [
            float(self.vocabulary.encode(token))
            for token in context
        ]

        target_id = float(
            self.vocabulary.encode(target)
        )

        return (
            input_ids,
            [target_id],
        )