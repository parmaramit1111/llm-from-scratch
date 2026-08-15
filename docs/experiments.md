# LLM From Scratch — Experiments

This document records the experiments used to understand and build the project from simple mathematical models toward a tiny language model.

Each experiment focuses on:

- What we are trying to learn
- What we implemented
- What happened
- What we learned
- What comes next

---

# Experiment 01 — Linear Learning

**Status:** Completed

## Objective

Teach the smallest possible model to discover:

```text
y = 3 × x
```

Training data:

```text
x    y
---------
1    3
2    6
3    9
4    12
5    15
```

## Model

```text
prediction = weight × input
```

Initial:

```text
weight = 0.5
learning rate = 0.01
epochs = 100
```

## Workflow

```text
Input
  ↓
Forward
  ↓
Prediction
  ↓
Loss
  ↓
Gradient
  ↓
Backpropagation
  ↓
Gradient Descent
  ↓
Updated Weight
  ↓
Repeat
```

## Result

```text
Initial weight: 0.5000
Final weight:   3.000000
Final loss:     0.000000
```

The model discovered the relationship without being given `weight = 3`.

## What We Learned

- A model can learn parameters from examples.
- Loss measures prediction error.
- Gradients indicate how parameters should change.
- Gradient descent updates parameters.
- Repeating the cycle is training.

---

# Experiment 02 — Linear Model Limitation

**Status:** Completed

## Objective

Show what happens when a model is too simple for the data.

Training data:

```text
x    y
---------
1    3
2    6
3    9
4    12
5    15
6    15
7    15
8    15
9    15
```

The model remains:

```text
prediction = weight × input
```

## Result

```text
Initial weight: 0.5000
Final weight:   1.585380
Final loss:     7.692325
```

The loss decreased but did not reach zero.

There is no single `weight` that can represent both:

```text
y = 3 × x
```

and the later constant value:

```text
y = 15
```

## What We Learned

The optimizer was not failing. The model was simply unable to represent the underlying relationship.

This introduced an important principle:

> A model can only learn relationships that its architecture can represent.

This motivates adding more expressive components.

---

# Experiment 03 — Add Bias

**Status:** Completed

## Objective

Extend the neuron from:

```text
y = 3 × x
```

to:

```text
y = 3 × x + 2
```

## Model

```text
prediction = (weight × input) + bias
```

Training data:

```text
x    y
---------
1    5
2    8
3    11
4    14
5    17
```

Initial parameters:

```text
weight = 0.5
bias   = 0.0
```

## New Concepts

- Bias as a trainable parameter
- Weight gradient
- Bias gradient
- Updating multiple parameters

Gradients:

```text
weight_gradient = output_gradient × input

bias_gradient = output_gradient
```

## Workflow

```text
Input
  ↓
Weight × Input + Bias
  ↓
Prediction
  ↓
Loss
  ↓
Loss Gradient
  ↓
Neuron Backward
  ├── Weight Gradient
  └── Bias Gradient
          ↓
      Optimizer
          ↓
Updated Weight + Bias
```

## What We Learned

A neuron can now learn both:

```text
weight
bias
```

The optimizer can update either parameter using the same:

```text
parameter + gradient + learning rate
```

mechanism.

---

# Experiment 04 — Multiple Neurons

**Status:** Completed

## Objective

Move from one neuron to a layer containing multiple neurons.

```text
              ┌── Neuron 1 ──→ Output 1
Input ────────┼── Neuron 2 ──→ Output 2
              └── Neuron 3 ──→ Output 3
```

Each neuron has its own:

```text
weight
bias
weight gradient
bias gradient
```

## New Concepts

- Layer
- Multiple neurons
- Multiple outputs
- Gradient propagation through a layer
- Multi-output loss
- Training a layer instead of a single neuron

## Experiment

The model was trained to learn:

```text
Neuron 1 → y = 3 × x
Neuron 2 → y = 5 × x
Neuron 3 → y = 7 × x
```

Starting weights:

```text
[0.5, 0.5, 0.5]
```

After extended training:

```text
[3.0, 5.0, 7.0]
```

with loss approaching zero.

## Training Flow

```text
Input
  ↓
Layer
  ├── Neuron 1
  ├── Neuron 2
  └── Neuron 3
  ↓
Predictions
  ↓
Multi-output Loss
  ↓
Gradients
  ↓
Layer Backward
  ↓
Each Neuron
  ↓
Optimizer
  ↓
Updated Parameters
```

## What We Learned

Multiple neurons can learn different relationships from the same input.

The `Layer` coordinates neurons, while each `Neuron` remains responsible for its own calculations and gradients.

---

# Experiment 05 — Activation Functions

**Status:** Completed

## Objective

Introduce non-linearity using ReLU.

ReLU stands for **Rectified Linear Unit**:

```text
ReLU(x) = max(0, x)
```

Forward behavior:

```text
positive → unchanged
negative → 0
zero     → 0
```

Backward behavior:

```text
raw input > 0  → gradient passes through
raw input ≤ 0  → gradient becomes 0
```

## Why Activation Matters

Without activation:

```text
Layer
  ↓
Linear calculation
  ↓
Layer
  ↓
Linear calculation
```

Stacking linear transformations still produces a linear transformation.

With activation:

```text
Layer
  ↓
Linear calculation
  ↓
ReLU
  ↓
Layer
  ↓
Linear calculation
  ↓
ReLU
```

the network can represent non-linear relationships.

## ReLU Experiment

Initial parameters:

```text
weight = 0.5
bias   = 1.0
```

Target relationship:

```text
output = ReLU(2 × input)
```

Training data:

```text
Input    Target
----------------
-3       0
-2       0
-1       0
 0       0
 1       2
 2       4
 3       6
```

After training:

```text
weight ≈ 2.0
bias   ≈ 0.0
loss   ≈ 0
```

Predictions:

```text
-3 → 0
-2 → 0
-1 → 0
 0 → 0
 1 → 2
 2 → 4
 3 → 6
```

## What We Learned

Activation functions transform the raw output of a neuron and introduce non-linearity.

ReLU also participates in backpropagation by controlling whether a gradient passes through the neuron.

---

# Experiment 06 — Training Loop

**Status:** Completed

## Objective

Create a reusable training component that coordinates the learning cycle.

The training step performs:

```text
Forward
  ↓
Loss
  ↓
Loss Gradient
  ↓
Backward
  ↓
Optimizer
  ↓
Updated Parameters
```

The experiment loop repeats this over:

```text
Epochs
  ↓
Training Examples
```

## Implementation

The training workflow is implemented in:

```text
src/llm_from_scratch/core/training.py
```

The trainer coordinates the model layer, loss function, and optimizer without implementing their internal mathematics.

---

# Experiment 07 — Numerical Gradient Check

**Status:** Planned

## Objective

Verify manually calculated gradients against numerical gradients.

Approximation:

```text
f(x + ε) - f(x - ε)
-------------------
        2ε
```

## Success Criteria

Analytical and numerical gradients should be very close.

This will give us confidence in our backpropagation implementation before the model becomes more complex.

---

# Experiment 08 — Character Prediction

**Status:** Planned

## Objective

Move from numerical relationships to text and predict the next character.

Example:

```text
Input     Target

h         e
he        l
hel       l
hell      o
```

## New Concepts

- Text data
- Character vocabulary
- Character IDs
- Sequence data
- Next-character prediction

---

# Experiment 09 — Character Tokenizer

**Status:** Planned

## Objective

Convert text into numerical token representations.

```text
"hello"
   ↓
[7, 4, 11, 11, 14]
```

and back:

```text
[7, 4, 11, 11, 14]
   ↓
"hello"
```

## New Concepts

- Vocabulary
- Token IDs
- Encode
- Decode
- Unknown tokens
- Special tokens

---

# Experiment 10 — Embeddings

**Status:** Planned

## Objective

Replace token IDs with learned vector representations.

```text
Token ID
   ↓
Embedding Matrix
   ↓
Vector
```

Example:

```text
drug
 ↓
[0.12, -0.42, 0.87, ...]
```

## Questions

- Why isn't an integer token ID enough?
- What does an embedding represent?
- How are embeddings learned?
- Do related tokens develop similar representations?

---

# Experiment 11 — Self-Attention

**Status:** Planned

## Objective

Implement single-head self-attention from scratch.

```text
Input Embeddings
       ↓
Query / Key / Value
       ↓
Attention Scores
       ↓
Weighted Values
       ↓
Context Representation
```

## Questions

- What do Query, Key, and Value represent?
- How are attention scores calculated?
- Why is scaling required?
- What information does attention capture?

---

# Experiment 12 — Multi-Head Attention

**Status:** Planned

## Objective

Extend self-attention to multiple heads.

```text
Input
  ↓
 ┌───────┬───────┬───────┐
Head 1  Head 2  Head 3  ...
 └───────┴───────┴───────┘
          ↓
        Combine
```

## Questions

- Why use multiple heads?
- Can different heads learn different relationships?
- What happens when the number of heads changes?

---

# Experiment 13 — Transformer Block

**Status:** Planned

## Objective

Build the first complete Transformer block.

```text
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
```

## New Concepts

- Residual connections
- Layer normalization
- Feed-forward networks
- Transformer blocks

---

# Experiment 14 — Tiny Transformer

**Status:** Planned

## Objective

Stack Transformer blocks and build a small language-model architecture.

```text
Text
 ↓
Tokenizer
 ↓
Token IDs
 ↓
Embeddings
 ↓
Positional Information
 ↓
Transformer Blocks
 ↓
Linear Output
 ↓
Logits
 ↓
Next Token
```

Training objective:

```text
Input:

"The pharmacy filled the"

Target:

"prescription"
```

---

# Experiment 15 — Tiny Language Model

**Status:** Planned

## Objective

Train a complete small language model on a controlled dataset.

Track:

```text
Parameter count
Dataset size
Training time
Epochs
Learning rate
Training loss
Validation loss
Generation examples
```

---

# Experiment 16 — Sampling

**Status:** Planned

## Objective

Understand how model probabilities become generated text.

Compare:

```text
Greedy decoding
Temperature
Top-K
Top-P
```

---

# Experiment 17 — Overfitting

**Status:** Planned

## Objective

Deliberately overfit a small dataset and observe:

```text
Training Loss ↓↓↓
Validation Loss ↑
```

Explore why overfitting happens and how it can be detected and reduced.

---

# Experiment 18 — NumPy Comparison

**Status:** Planned

## Objective

Compare selected components implemented with pure Python and NumPy.

Measure:

```text
Execution time
Memory usage
Implementation complexity
```

---

# Experiment 19 — PyTorch Comparison

**Status:** Planned

## Objective

Implement the same basic model using PyTorch and compare it with our implementation.

The goal is to understand what a framework such as PyTorch abstracts away.

---

# Experiment 20 — C++ Implementation

**Status:** Future

## Objective

Move a computationally important component to C++.

Initial candidate:

```text
Matrix multiplication
```

Compare:

```text
Python
NumPy
C++
```

---

# Experiment 21 — Rust Implementation

**Status:** Future

## Objective

Implement a selected performance-critical component in Rust.

Explore:

- Ownership
- Borrowing
- Memory management
- Concurrency
- Performance
- Python/Rust interoperability

---

# Experiment 22 — GPU Exploration

**Status:** Future

## Objective

Understand why GPUs accelerate neural-network workloads.

```text
CPU
 ↓
GPU
 ↓
Parallel Computation
 ↓
Matrix Operations
```

Later investigate CUDA and GPU kernels.

---

# Experiment Rules

## Rule 1 — Record failures

A failed experiment is useful if we record:

```text
What failed
Why it failed
How it was fixed
What we learned
```

## Rule 2 — Change one important variable at a time

When investigating behavior, avoid changing the architecture, dataset, learning rate, and optimizer simultaneously.

## Rule 3 — Keep experiments reproducible

Record configuration, dataset, and random seed whenever practical.

## Rule 4 — Measure before optimizing

Do not assume something is slow. Measure it.

## Rule 5 — Understand before using frameworks

Understand the underlying concept before introducing a framework abstraction.

```text
Manual Gradient
      ↓
Automatic Differentiation

Manual Matrix Operations
      ↓
NumPy

Manual Neural Network
      ↓
PyTorch

Manual Attention
      ↓
Transformer Implementation
```

---

# Current Progress

Completed:

```text
01 — Linear Learning             ✅
02 — Linear Model Limitation     ✅
03 — Add Bias                    ✅
04 — Multiple Neurons            ✅
05 — Activation Functions        ✅
06 — Training Loop               ✅
```

Current architecture:

```text
Input
  ↓
Layer
  ↓
Multiple Neurons
  ↓
Weight × Input + Bias
  ↓
Activation
  ↓
Output
  ↓
Loss
  ↓
Backpropagation
  ↓
Gradient Descent
  ↓
Updated Parameters
```

The project has now demonstrated:

```text
Neuron
  ↓
Layer
  ↓
Activation
  ↓
Loss
  ↓
Backpropagation
  ↓
Optimizer
  ↓
Training
```

---

# Immediate Next Step

The next major capability is:

```text
Multiple Layers
```

The target architecture is:

```text
Input
  ↓
Layer 1
  ↓
Activation
  ↓
Layer 2
  ↓
Activation
  ↓
Output
```

This will introduce forward and backward propagation across multiple layers and move the project toward the architecture used by modern neural networks.

After that:

```text
Character Prediction
  ↓
Tokenizer
  ↓
Embeddings
  ↓
Self-Attention
  ↓
Transformer
  ↓
Tiny Language Model
```

---

# Long-Term Goal

Build a tiny language model from first principles.

The important outcome is not the size of the final model.

The important outcome is:

> **Understanding how the model learns and how each component contributes to a language model.**
