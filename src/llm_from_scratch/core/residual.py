"""
Concept: Residual Connection

A residual connection adds the input of a sublayer
to the output of that sublayer.

    output = input + sublayer_output

During backward propagation, the gradient flows
directly through both paths.
"""


class Residual:
    """Add a residual connection."""

    def forward(
        self,
        inputs: list[list[float]],
        sublayer_output: list[list[float]],
    ) -> list[list[float]]:
        """Add the input to the sublayer output."""

        return [
            [
                input_value + output_value
                for input_value, output_value in zip(
                    input_row,
                    output_row,
                    strict=True,
                )
            ]
            for input_row, output_row in zip(
                inputs,
                sublayer_output,
                strict=True,
            )
        ]

    def backward(
        self,
        output_gradients: list[list[float]],
    ) -> tuple[
        list[list[float]],
        list[list[float]],
    ]:
        """Propagate gradients through the residual connection."""

        input_gradients = [
            row.copy()
            for row in output_gradients
        ]

        sublayer_gradients = [
            row.copy()
            for row in output_gradients
        ]

        return (
            input_gradients,
            sublayer_gradients,
        )