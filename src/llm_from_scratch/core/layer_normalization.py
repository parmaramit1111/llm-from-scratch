"""
Concept: Layer Normalization

Layer normalization normalizes each input row independently.

    mean = average(x)

    variance = average((x - mean)²)

    normalized = (x - mean) / √(variance + ε)

    output = γ × normalized + β
"""


import math


class LayerNormalization:
    """Normalize each input row independently."""

    def __init__(
        self,
        normalized_size: int,
        epsilon: float = 1e-5,
    ):
        self.normalized_size = normalized_size
        self.epsilon = epsilon

        # Trainable parameters.
        self.gamma = [
            1.0
            for _ in range(normalized_size)
        ]

        self.beta = [
            0.0
            for _ in range(normalized_size)
        ]

        # Values cached during the forward pass.
        self.inputs: list[list[float]] = []
        self.means: list[float] = []
        self.variances: list[float] = []
        self.normalized: list[list[float]] = []
        self.outputs: list[list[float]] = []

        self.gamma_gradients: list[float] = []
        self.beta_gradients: list[float] = []
        self.input_gradients: list[list[float]] = []

    def forward(
        self,
        inputs: list[list[float]],
    ) -> list[list[float]]:
        """Normalize each input row."""

        self.inputs = inputs

        self.means = []
        self.variances = []
        self.normalized = []
        self.outputs = []

        for row in inputs:
            mean = sum(row) / self.normalized_size

            variance = sum(
                (value - mean) ** 2
                for value in row
            ) / self.normalized_size

            normalized_row = [
                (value - mean)
                / math.sqrt(variance + self.epsilon)
                for value in row
            ]

            output_row = [
                self.gamma[index] * normalized_row[index]
                + self.beta[index]
                for index in range(self.normalized_size)
            ]

            self.means.append(mean)
            self.variances.append(variance)
            self.normalized.append(normalized_row)
            self.outputs.append(output_row)

        return self.outputs

    def backward(
        self,
        output_gradients: list[list[float]],
    ) -> list[list[float]]:
        """Propagate gradients through layer normalization."""

        # ---------------------------------------------------------
        # output = gamma × normalized + beta
        #
        # dGamma = sum(dOutput × normalized)
        # dBeta  = sum(dOutput)
        # ---------------------------------------------------------

        self.gamma_gradients = [
            sum(
                output_gradients[row][column]
                * self.normalized[row][column]
                for row in range(len(self.inputs))
            )
            for column in range(self.normalized_size)
        ]

        self.beta_gradients = [
            sum(
                output_gradients[row][column]
                for row in range(len(self.inputs))
            )
            for column in range(self.normalized_size)
        ]

        # ---------------------------------------------------------
        # Backward through normalization.
        #
        # For each row:
        #
        # y = gamma × x_normalized + beta
        #
        # dx_normalized = dy × gamma
        #
        # dx = (1 / sqrt(var + epsilon)) ×
        #      (dx_normalized
        #       - mean(dx_normalized)
        #       - x_normalized × mean(dx_normalized × x_normalized))
        # ---------------------------------------------------------

        self.input_gradients = []

        for row_index, row in enumerate(self.inputs):
            gradients = [
                output_gradients[row_index][column]
                * self.gamma[column]
                for column in range(self.normalized_size)
            ]

            mean_gradient = (
                sum(gradients)
                / self.normalized_size
            )

            normalized_gradient_mean = (
                sum(
                    gradients[column]
                    * self.normalized[row_index][column]
                    for column in range(self.normalized_size)
                )
                / self.normalized_size
            )

            scale = 1.0 / math.sqrt(
                self.variances[row_index]
                + self.epsilon
            )

            input_gradient_row = [
                scale
                * (
                    gradients[column]
                    - mean_gradient
                    - self.normalized[row_index][column]
                    * normalized_gradient_mean
                )
                for column in range(self.normalized_size)
            ]

            self.input_gradients.append(
                input_gradient_row
            )

        return self.input_gradients