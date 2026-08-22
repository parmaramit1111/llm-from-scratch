from llm_from_scratch.core.softmax import Softmax


def test_softmax_probabilities_sum_to_one():
    softmax = Softmax()

    probabilities = softmax.forward(
        logits=[1.0, 2.0, 3.0],
    )

    assert abs(sum(probabilities) - 1.0) < 1e-10


def test_softmax_probabilities_are_between_zero_and_one():
    softmax = Softmax()

    probabilities = softmax.forward(
        logits=[1.0, 2.0, 3.0],
    )

    assert all(
        0.0 <= probability <= 1.0
        for probability in probabilities
    )


def test_larger_logit_gets_higher_probability():
    softmax = Softmax()

    probabilities = softmax.forward(
        logits=[1.0, 2.0, 3.0],
    )

    assert probabilities[2] > probabilities[1]
    assert probabilities[1] > probabilities[0]


def test_softmax_is_numerically_stable():
    softmax = Softmax()

    probabilities = softmax.forward(
        logits=[1000.0, 1001.0, 1002.0],
    )

    assert abs(sum(probabilities) - 1.0) < 1e-10