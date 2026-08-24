"""
Concept: Feed-Forward Network

A Transformer feed-forward network applies two linear
transformations with a non-linear activation between them.

    hidden = ReLU(X × W1 + b1)

    output = hidden × W2 + b2
"""


class FeedForward:
    """Calculate a two-layer feed-forward network."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # First projection.
        self.input_weights = [
            [0.1 for _ in range(hidden_size)]
            for _ in range(input_size)
        ]

        self.input_biases = [
            0.0
            for _ in range(hidden_size)
        ]

        # Second projection.
        self.output_weights = [
            [0.1 for _ in range(output_size)]
            for _ in range(hidden_size)
        ]

        self.output_biases = [
            0.0
            for _ in range(output_size)
        ]

        self.inputs: list[list[float]] = []
        self.hidden: list[list[float]] = []
        self.outputs: list[list[float]] = []

        self.hidden_linear: list[list[float]] = []

        self.input_weight_gradients: list[list[float]] = []
        self.input_bias_gradients: list[float] = []

        self.output_weight_gradients: list[list[float]] = []
        self.output_bias_gradients: list[float] = []

        self.input_gradients: list[list[float]] = []

    def _relu(
        self,
        value: float,
    ) -> float:
        """Apply the ReLU activation."""

        return max(0.0, value)

    def forward(
        self,
        inputs: list[list[float]],
    ) -> list[list[float]]:
        """Calculate the feed-forward output."""

        self.inputs = inputs

        # First linear transformation.
        self.hidden_linear = [
            [
                sum(
                    inputs[row][index]
                    * self.input_weights[index][column]
                    for index in range(self.input_size)
                )
                + self.input_biases[column]
                for column in range(self.hidden_size)
            ]
            for row in range(len(inputs))
        ]

        # ReLU activation.
        self.hidden = [
            [
                self._relu(value)
                for value in row
            ]
            for row in self.hidden_linear
        ]

        # Second linear transformation.
        self.outputs = [
            [
                sum(
                    self.hidden[row][index]
                    * self.output_weights[index][column]
                    for index in range(self.hidden_size)
                )
                + self.output_biases[column]
                for column in range(self.output_size)
            ]
            for row in range(len(inputs))
        ]

        return self.outputs

    def backward(
        self,
        output_gradients: list[list[float]],
    ) -> list[list[float]]:
        """Propagate gradients through the feed-forward network."""

        # ---------------------------------------------------------
        # Second linear layer
        #
        # output = hidden × W2 + b2
        #
        # dHidden = dOutput × W2ᵀ
        # ---------------------------------------------------------

        hidden_gradients = [
            [
                sum(
                    output_gradients[row][column]
                    * self.output_weights[index][column]
                    for column in range(self.output_size)
                )
                for index in range(self.hidden_size)
            ]
            for row in range(len(self.inputs))
        ]

        # ---------------------------------------------------------
        # ReLU
        # ---------------------------------------------------------

        hidden_linear_gradients = [
            [
                gradient
                if self.hidden_linear[row][column] > 0.0
                else 0.0
                for column, gradient in enumerate(
                    hidden_gradients[row]
                )
            ]
            for row in range(len(self.inputs))
        ]

        # ---------------------------------------------------------
        # Second-layer parameter gradients
        #
        # dW2 = hiddenᵀ × dOutput
        # db2 = sum(dOutput)
        # ---------------------------------------------------------

        self.output_weight_gradients = [
            [
                sum(
                    self.hidden[row][index]
                    * output_gradients[row][column]
                    for row in range(len(self.inputs))
                )
                for column in range(self.output_size)
            ]
            for index in range(self.hidden_size)
        ]

        self.output_bias_gradients = [
            sum(
                output_gradients[row][column]
                for row in range(len(self.inputs))
            )
            for column in range(self.output_size)
        ]

        # ---------------------------------------------------------
        # First linear layer
        #
        # dInput = dHiddenLinear × W1ᵀ
        # ---------------------------------------------------------

        self.input_gradients = [
            [
                sum(
                    hidden_linear_gradients[row][index]
                    * self.input_weights[column][index]
                    for index in range(self.hidden_size)
                )
                for column in range(self.input_size)
            ]
            for row in range(len(self.inputs))
        ]

        # ---------------------------------------------------------
        # First-layer parameter gradients
        #
        # dW1 = inputᵀ × dHiddenLinear
        # db1 = sum(dHiddenLinear)
        # ---------------------------------------------------------

        self.input_weight_gradients = [
            [
                sum(
                    self.inputs[row][index]
                    * hidden_linear_gradients[row][column]
                    for row in range(len(self.inputs))
                )
                for column in range(self.hidden_size)
            ]
            for index in range(self.input_size)
        ]

        self.input_bias_gradients = [
            sum(
                hidden_linear_gradients[row][column]
                for row in range(len(self.inputs))
            )
            for column in range(self.hidden_size)
        ]

        return self.input_gradients