"""
Concept: Self-Attention

Self-attention allows each token in a sequence to determine
how much attention it should give to other tokens.

The basic calculation is:

    Q = X × Wq
    K = X × Wk
    V = X × Wv

    scores = Q × Kᵀ

    scaled_scores = scores / √d

    attention_weights = softmax(scaled_scores)

    output = attention_weights × V
"""


import math


class SelfAttention:
    """Calculate scaled dot-product self-attention."""

    def __init__(
        self,
        input_size: int,
        attention_size: int,
    ):
        self.input_size = input_size
        self.attention_size = attention_size

        # Projection matrices.
        self.query_weights = [
            [0.1 for _ in range(attention_size)]
            for _ in range(input_size)
        ]

        self.key_weights = [
            [0.1 for _ in range(attention_size)]
            for _ in range(input_size)
        ]

        self.value_weights = [
            [0.1 for _ in range(attention_size)]
            for _ in range(input_size)
        ]

        self.attention_weights: list[list[float]] = []

        self.inputs: list[list[float]] = []
        self.queries: list[list[float]] = []
        self.keys: list[list[float]] = []
        self.values: list[list[float]] = []
        self.scores: list[list[float]] = []
        self.scaled_scores: list[list[float]] = []
        self.attention_weight_gradients: list[list[float]] = []
        self.value_gradients: list[list[float]] = []
        self.scaled_score_gradients: list[list[float]] = []
        self.score_gradients: list[list[float]] = []
        self.query_gradients: list[list[float]] = []
        self.key_gradients: list[list[float]] = []

        self.query_weight_gradients: list[list[float]] = []
        self.key_weight_gradients: list[list[float]] = []
        self.value_weight_gradients: list[list[float]] = []


    def _matmul(
        self,
        left: list[list[float]],
        right: list[list[float]],
    ) -> list[list[float]]:
        """Multiply two matrices."""

        return [
            [
                sum(
                    left[row][index] * right[index][column]
                    for index in range(len(right))
                )
                for column in range(len(right[0]))
            ]
            for row in range(len(left))
        ]

    def _transpose(
        self,
        matrix: list[list[float]],
    ) -> list[list[float]]:
        """Transpose a matrix."""

        return [
            [
                matrix[row][column]
                for row in range(len(matrix))
            ]
            for column in range(len(matrix[0]))
        ]

    def _softmax(
        self,
        values: list[float],
    ) -> list[float]:
        """Calculate a numerically stable softmax."""

        maximum = max(values)

        exponentials = [
            math.exp(value - maximum)
            for value in values
        ]

        total = sum(exponentials)

        return [
            value / total
            for value in exponentials
        ]

    def _softmax_backward(
        self,
        probabilities: list[float],
        gradients: list[float],
    ) -> list[float]:
        """Calculate the gradient through softmax."""

        dot_product = sum(
            gradient * probability
            for gradient, probability in zip(
                gradients,
                probabilities,
                strict=True,
            )
        )

        return [
            probability * (gradient - dot_product)
            for probability, gradient in zip(
                probabilities,
                gradients,
                strict=True,
            )
        ]

    def forward(
        self,
        inputs: list[list[float]],
    ) -> list[list[float]]:
        """Calculate the self-attention output."""

        self.inputs = inputs

        # Q = X × Wq
        queries = self._matmul(
            inputs,
            self.query_weights,
        )

        # K = X × Wk
        keys = self._matmul(
            inputs,
            self.key_weights,
        )

        # V = X × Wv
        values = self._matmul(
            inputs,
            self.value_weights,
        )

        self.queries = queries
        self.keys = keys
        self.values = values

        # Q × Kᵀ
        scores = self._matmul(
            queries,
            self._transpose(keys),
        )

        self.scores = scores

        # Scale scores by √d.
        scale = math.sqrt(self.attention_size)

        scaled_scores = [
            [
                score / scale
                for score in row
            ]
            for row in scores
        ]

        self.scaled_scores = scaled_scores

        # Convert each score row into probabilities.
        attention_weights = [
            self._softmax(row)
            for row in scaled_scores
        ]

        self.attention_weights = attention_weights

        # Weighted values.
        output = self._matmul(
            attention_weights,
            values,
        )

        return output

    def backward(
        self,
        output_gradients: list[list[float]],
    ) -> list[list[float]]:
        """Propagate gradients through the attention output."""

        # Output:
        #
        #     O = A × V
        #
        # Therefore:
        #
        #     dA = dO × Vᵀ
        #     dV = Aᵀ × dO

        attention_weight_gradients = self._matmul(
            output_gradients,
            self._transpose(self.values),
        )

        value_gradients = self._matmul(
            self._transpose(self.attention_weights),
            output_gradients,
        )

        # Store the gradients for the next backward step.
        self.attention_weight_gradients = (
            attention_weight_gradients
        )

        self.value_gradients = value_gradients

        scaled_score_gradients = [
            self._softmax_backward(
                probabilities=attention_weights,
                gradients=gradient_row,
            )
            for attention_weights, gradient_row in zip(
                self.attention_weights,
                self.attention_weight_gradients,
                strict=True,
            )
        ]

        self.scaled_score_gradients = scaled_score_gradients

        score_gradients = [
            [
                gradient / math.sqrt(self.attention_size)
                for gradient in row
            ]
            for row in self.scaled_score_gradients
        ]

        self.score_gradients = score_gradients

        query_gradients = self._matmul(
            self.score_gradients,
            self.keys,
        )

        key_gradients = self._matmul(
            self._transpose(self.score_gradients),
            self.queries,
        )

        self.query_gradients = query_gradients
        self.key_gradients = key_gradients

        query_weight_gradients = self._matmul(
            self._transpose(self.inputs),
            self.query_gradients,
        )

        key_weight_gradients = self._matmul(
            self._transpose(self.inputs),
            self.key_gradients,
        )

        value_weight_gradients = self._matmul(
            self._transpose(self.inputs),
            self.value_gradients,
        )

        self.query_weight_gradients = query_weight_gradients
        self.key_weight_gradients = key_weight_gradients
        self.value_weight_gradients = value_weight_gradients

        input_gradients_from_query = self._matmul(
            self.query_gradients,
            self._transpose(self.query_weights),
        )

        input_gradients_from_key = self._matmul(
            self.key_gradients,
            self._transpose(self.key_weights),
        )

        input_gradients_from_value = self._matmul(
            self.value_gradients,
            self._transpose(self.value_weights),
        )


        # Combine gradients from the Q, K, and V paths.
        input_gradients = [
            [
                query_gradient
                + key_gradient
                + value_gradient
                for query_gradient, key_gradient, value_gradient in zip(
                    query_row,
                    key_row,
                    value_row,
                    strict=True,
                )
            ]
            for query_row, key_row, value_row in zip(
                input_gradients_from_query,
                input_gradients_from_key,
                input_gradients_from_value,
                strict=True,
            )
        ]

        return input_gradients