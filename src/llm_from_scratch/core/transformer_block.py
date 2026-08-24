"""
Concept: Transformer Block

A Transformer block combines:

    Input
      ↓
    Self-Attention
      ↓
    Residual Connection
      ↓
    Layer Normalization
      ↓
    Feed-Forward Network
      ↓
    Residual Connection
      ↓
    Layer Normalization
      ↓
    Output
"""

from llm_from_scratch.core.attention import SelfAttention
from llm_from_scratch.core.feed_forward import FeedForward
from llm_from_scratch.core.layer_normalization import (
    LayerNormalization,
)
from llm_from_scratch.core.residual import Residual


class TransformerBlock:
    """Build a complete Transformer block."""

    def __init__(
        self,
        input_size: int,
        attention_size: int,
        feed_forward_hidden_size: int,
    ):
        self.input_size = input_size
        self.attention_size = attention_size
        self.feed_forward_hidden_size = (
            feed_forward_hidden_size
        )

        # ---------------------------------------------------------
        # Self-Attention
        # ---------------------------------------------------------

        self.attention = SelfAttention(
            input_size=input_size,
            attention_size=attention_size,
        )

        # ---------------------------------------------------------
        # First residual connection + normalization
        # ---------------------------------------------------------

        self.attention_residual = Residual()

        self.attention_layer_normalization = (
            LayerNormalization(
                normalized_size=attention_size,
            )
        )

        # ---------------------------------------------------------
        # Feed-Forward Network
        # ---------------------------------------------------------

        self.feed_forward = FeedForward(
            input_size=attention_size,
            hidden_size=feed_forward_hidden_size,
            output_size=attention_size,
        )

        # ---------------------------------------------------------
        # Second residual connection + normalization
        # ---------------------------------------------------------

        self.feed_forward_residual = Residual()

        self.feed_forward_layer_normalization = (
            LayerNormalization(
                normalized_size=attention_size,
            )
        )

        # ---------------------------------------------------------
        # Forward-pass state
        # ---------------------------------------------------------

        self.inputs: list[list[float]] = []

        self.attention_output: list[list[float]] = []
        self.attention_residual_output: list[list[float]] = []
        self.attention_normalized_output: list[list[float]] = []

        self.feed_forward_output: list[list[float]] = []
        self.feed_forward_residual_output: list[list[float]] = []

        self.outputs: list[list[float]] = []

        # ---------------------------------------------------------
        # Backward-pass state
        #
        # The first-half backward fields are retained from the
        # previous implementation. Additional fields will be used
        # when we implement the complete backward path.
        # ---------------------------------------------------------

        self.layer_normalization_gradients: (
            list[list[float]]
        ) = []

        self.residual_input_gradients: (
            list[list[float]]
        ) = []

        self.residual_sublayer_gradients: (
            list[list[float]]
        ) = []

        self.input_gradients: list[list[float]] = []

        self.attention_normalization_gradients: (
            list[list[float]]
        ) = []

        self.attention_residual_input_gradients: (
            list[list[float]]
        ) = []

        self.attention_residual_sublayer_gradients: (
            list[list[float]]
        ) = []

        self.feed_forward_input_gradients: (
            list[list[float]]
        ) = []

        self.feed_forward_residual_input_gradients: (
            list[list[float]]
        ) = []

        self.feed_forward_residual_sublayer_gradients: (
            list[list[float]]
        ) = []

    def forward(
        self,
        inputs: list[list[float]],
    ) -> list[list[float]]:
        """Run the complete Transformer block."""

        self.inputs = inputs

        # ---------------------------------------------------------
        # Self-Attention
        # ---------------------------------------------------------

        self.attention_output = self.attention.forward(
            inputs,
        )

        # ---------------------------------------------------------
        # First Residual Connection
        #
        # attention_residual =
        #     input + attention_output
        # ---------------------------------------------------------

        self.attention_residual_output = (
            self.attention_residual.forward(
                inputs,
                self.attention_output,
            )
        )

        # ---------------------------------------------------------
        # First Layer Normalization
        # ---------------------------------------------------------

        self.attention_normalized_output = (
            self.attention_layer_normalization.forward(
                self.attention_residual_output,
            )
        )

        # ---------------------------------------------------------
        # Feed-Forward Network
        # ---------------------------------------------------------

        self.feed_forward_output = (
            self.feed_forward.forward(
                self.attention_normalized_output,
            )
        )

        # ---------------------------------------------------------
        # Second Residual Connection
        #
        # feed_forward_residual =
        #     attention_normalized_output
        #     + feed_forward_output
        # ---------------------------------------------------------

        self.feed_forward_residual_output = (
            self.feed_forward_residual.forward(
                self.attention_normalized_output,
                self.feed_forward_output,
            )
        )

        # ---------------------------------------------------------
        # Second Layer Normalization
        # ---------------------------------------------------------

        self.outputs = (
            self.feed_forward_layer_normalization.forward(
                self.feed_forward_residual_output,
            )
        )

        return self.outputs

    def backward(
        self,
        output_gradients: list[list[float]],
    ) -> list[list[float]]:
        """Propagate gradients through the complete Transformer block."""

        # ---------------------------------------------------------
        # Second Layer Normalization
        # ---------------------------------------------------------
        #
        # feed_forward_residual_output
        #     ↓
        # LayerNorm #2
        #     ↓
        # output
        #
        # Therefore:
        #
        # dFeedForwardResidual =
        #     dOutput passed through LayerNorm #2
        # ---------------------------------------------------------

        feed_forward_residual_gradients = (
            self.feed_forward_layer_normalization.backward(
                output_gradients,
            )
        )

        # ---------------------------------------------------------
        # Second Residual Connection
        #
        # feed_forward_residual =
        #     attention_normalized_output
        #     + feed_forward_output
        #
        # The gradient flows through both paths.
        # ---------------------------------------------------------

        (
            feed_forward_residual_input_gradients,
            feed_forward_residual_sublayer_gradients,
        ) = self.feed_forward_residual.backward(
            feed_forward_residual_gradients,
        )

        self.feed_forward_residual_input_gradients = (
            feed_forward_residual_input_gradients
        )

        self.feed_forward_residual_sublayer_gradients = (
            feed_forward_residual_sublayer_gradients
        )

        # ---------------------------------------------------------
        # Feed-Forward Network
        #
        # The residual sublayer path flows backward through
        # the Feed-Forward Network.
        # ---------------------------------------------------------

        feed_forward_input_gradients = (
            self.feed_forward.backward(
                feed_forward_residual_sublayer_gradients,
            )
        )

        self.feed_forward_input_gradients = (
            feed_forward_input_gradients
        )

        # ---------------------------------------------------------
        # Merge the two paths going into LayerNorm #1
        #
        # Path 1:
        #
        #     Residual #2
        #         ↓
        #     attention_normalized_output
        #
        # Path 2:
        #
        #     Residual #2
        #         ↓
        #     FeedForward
        #         ↓
        #     attention_normalized_output
        #
        # Therefore:
        #
        # dAttentionNormalized =
        #     dResidualInput
        #     + dFeedForwardInput
        # ---------------------------------------------------------

        attention_normalized_gradients = [
            [
                residual_gradient + feed_forward_gradient
                for residual_gradient, feed_forward_gradient in zip(
                    residual_row,
                    feed_forward_row,
                    strict=True,
                )
            ]
            for residual_row, feed_forward_row in zip(
                feed_forward_residual_input_gradients,
                feed_forward_input_gradients,
                strict=True,
            )
        ]

        # ---------------------------------------------------------
        # First Layer Normalization
        # ---------------------------------------------------------

        attention_residual_gradients = (
            self.attention_layer_normalization.backward(
                attention_normalized_gradients,
            )
        )

        self.layer_normalization_gradients = (
            attention_residual_gradients
        )

        # ---------------------------------------------------------
        # First Residual Connection
        #
        # attention_residual =
        #     input + attention_output
        # ---------------------------------------------------------

        (
            attention_residual_input_gradients,
            attention_residual_sublayer_gradients,
        ) = self.attention_residual.backward(
            attention_residual_gradients,
        )

        self.attention_residual_input_gradients = (
            attention_residual_input_gradients
        )

        self.attention_residual_sublayer_gradients = (
            attention_residual_sublayer_gradients
        )

        # ---------------------------------------------------------
        # Self-Attention
        # ---------------------------------------------------------

        attention_input_gradients = (
            self.attention.backward(
                attention_residual_sublayer_gradients,
            )
        )

        # ---------------------------------------------------------
        # Final input gradient
        #
        # The original input receives gradients from:
        #
        # 1. First residual's direct input path
        # 2. Self-attention path
        # ---------------------------------------------------------

        self.input_gradients = [
            [
                residual_gradient + attention_gradient
                for residual_gradient, attention_gradient in zip(
                    residual_row,
                    attention_row,
                    strict=True,
                )
            ]
            for residual_row, attention_row in zip(
                attention_residual_input_gradients,
                attention_input_gradients,
                strict=True,
            )
        ]

        return self.input_gradients