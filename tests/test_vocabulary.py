from llm_from_scratch.language.vocabulary import Vocabulary


def test_vocabulary_encodes_tokens():
    vocabulary = Vocabulary(["h", "e", "l", "o"])

    assert vocabulary.encode("h") == 0
    assert vocabulary.encode("e") == 1
    assert vocabulary.encode("l") == 2
    assert vocabulary.encode("o") == 3


def test_vocabulary_decodes_ids():
    vocabulary = Vocabulary(["h", "e", "l", "o"])

    assert vocabulary.decode(0) == "h"
    assert vocabulary.decode(1) == "e"
    assert vocabulary.decode(2) == "l"
    assert vocabulary.decode(3) == "o"


def test_vocabulary_size():
    vocabulary = Vocabulary(["h", "e", "l", "o"])

    assert vocabulary.size == 4