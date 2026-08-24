"""
Concept: Vocabulary

A vocabulary maps tokens to integer IDs and integer IDs back to tokens.

For the first language-model experiments, each token will be a character.

Example:

    h → 0
    e → 1
    l → 2
    o → 3

The reverse mapping is also supported:

    0 → h
    1 → e
    2 → l
    3 → o
"""


class Vocabulary:
    """Map tokens to IDs and IDs back to tokens."""

    def __init__(self, tokens: list[str]):
        self.token_to_id = {
            token: index
            for index, token in enumerate(tokens)
        }

        self.id_to_token = {
            index: token
            for index, token in enumerate(tokens)
        }

    @property
    def size(self) -> int:
        """Return the number of tokens in the vocabulary."""
        return len(self.token_to_id)

    def encode(self, token: str) -> int:
        """Convert a token into its integer ID."""
        return self.token_to_id[token]

    def decode(self, token_id: int) -> str:
        """Convert an integer ID back into its token."""
        return self.id_to_token[token_id]