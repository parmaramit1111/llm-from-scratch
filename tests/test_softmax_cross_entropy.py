from llm_from_scratch.core.softmax_cross_entropy import (
    SoftmaxCrossEntropy,
)
from llm_from_scratch.core.gradient_check import GradientChecker


def test_softmax_cross_entropy_forward():
    loss_function = SoftmaxCrossEntropy()

    loss = loss_function.forward(
        logits=[1.0, 2.0, 3.0],
        target_index=2,
    )

    assert abs(loss - 0.4076059644) < 1e-10


def test_softmax_cross_entropy_backward():
    loss_function = SoftmaxCrossEntropy()

    gradients = loss_function.backward(
        logits=[1.0, 2.0, 3.0],
        target_index=2,
    )

    assert abs(gradients[0] - 0.0900305732) < 1e-10
    assert abs(gradients[1] - 0.2447284711) < 1e-10
    assert abs(gradients[2] - (-0.3347590443)) < 1e-10


def test_softmax_cross_entropy_gradient():
    """Analytical logits gradients should match numerical gradients."""

    logits = [1.0, 2.0, 3.0]
    target_index = 2

    loss_function = SoftmaxCrossEntropy()

    analytical_gradients = loss_function.backward(
        logits=logits,
        target_index=target_index,
    )

    checker = GradientChecker(
        tolerance=1e-5,
    )

    for index, analytical_gradient in enumerate(
        analytical_gradients
    ):
        def loss_for_logit(value: float) -> float:
            test_logits = logits.copy()
            test_logits[index] = value

            return loss_function.forward(
                logits=test_logits,
                target_index=target_index,
            )

        numerical_gradient = checker.numerical_gradient(
            parameter=logits[index],
            loss_function=loss_for_logit,
        )

        assert checker.check(
            analytical_gradient=analytical_gradient,
            numerical_gradient=numerical_gradient,
        )