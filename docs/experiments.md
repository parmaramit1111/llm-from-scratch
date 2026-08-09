# LLM From Scratch — Experiments

This document records the practical experiments performed during the development of the project.

The purpose is not only to record successful results, but also to document:

* What we tried
* Why we tried it
* What happened
* What failed
* What we learned
* What we will change next

The experiments should progressively move from simple mathematical models toward a tiny language model.

---

# Experiment 01 — Linear Learning

**Status:** Planned

## Objective

Build the smallest possible learning system and teach it to discover the relationship:

```text
y = 3x
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

The model should start with a random weight rather than being given the value `3`.

---

## Model

Start with:

```text
prediction = weight × input
```

Initially:

```text
weight = random value
```

The model should gradually learn:

```text
weight ≈ 3
```

---

## Learning Workflow

```text
Input
  ↓
Forward Pass
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

---

## Implementation

Initial files:

```text
src/llm_from_scratch/core/
├── neuron.py
├── loss.py
├── optimizer.py
└── training.py

experiments/
└── 01_linear_learning.py
```

Tests:

```text
tests/
├── test_neuron.py
└── test_loss.py
```

---

## Initial Parameters

To be recorded after implementation:

```text
Initial weight:
Learning rate:
Epochs:
Training examples:
```

---

## Results

To be filled after the experiment:

```text
Final weight:
Final loss:
Training time:
Number of iterations:
```

Expected behavior:

```text
Loss should decrease
Weight should approach 3
Predictions should approach the target values
```

---

## Observations

To be recorded during the experiment.

Questions to answer:

* Did the loss decrease smoothly?
* How quickly did the weight approach 3?
* What happens if the learning rate is too small?
* What happens if the learning rate is too large?
* What happens if the initial weight is negative?
* Does the model converge from different starting weights?

---

## What We Learned

To be completed after the experiment.

The goal is to explain in our own words:

```text
Forward pass
Loss
Derivative
Gradient
Backpropagation
Gradient descent
Learning rate
Parameter update
```

---

# Experiment 02 — Add Bias

**Status:** Planned

## Objective

Extend the model from:

```text
y = 3x
```

to:

```text
y = 3x + 2
```

Model:

```text
prediction = weight × input + bias
```

The model must learn both:

```text
weight ≈ 3
bias ≈ 2
```

---

## New Concepts

* Bias
* Multiple trainable parameters
* Multiple gradients
* Parameter updates

---

## Questions

* Does adding bias improve the model's ability to represent the data?
* Can both parameters converge?
* How does the gradient differ for weight and bias?

---

# Experiment 03 — Multiple Neurons

**Status:** Planned

## Objective

Move from a single neuron to a small layer.

```text
Input
  ↓
Neuron 1
Neuron 2
Neuron 3
  ↓
Output
```

---

## New Concepts

* Layers
* Multiple weights
* Multiple biases
* Vectors
* Matrix operations

---

## Questions

* How do multiple neurons cooperate?
* What does each neuron learn?
* Does adding neurons improve model capacity?

---

# Experiment 04 — Activation Functions

**Status:** Planned

## Objective

Introduce non-linearity.

Initial activation:

```text
ReLU
```

Later:

```text
Sigmoid
Tanh
GELU
```

---

## Experiment

Train a model on a relationship that cannot be represented well using a simple linear function.

Compare:

```text
Without activation
vs
With activation
```

---

## Questions

* Why do neural networks need non-linear activation?
* What happens when all layers are linear?
* How does ReLU change the output?
* What happens to gradients?

---

# Experiment 05 — Training Loop

**Status:** Planned

## Objective

Build a reusable training loop.

Conceptually:

```python
for epoch in epochs:

    prediction = model.forward(input)

    loss = loss_function(prediction, target)

    gradient = loss_function.backward(
        prediction,
        target
    )

    model.backward(gradient)

    optimizer.step(
        model.parameters()
    )
```

---

## Metrics

Track:

```text
Epoch
Loss
Parameters
Training time
```

Example:

```text
Epoch    Loss
----------------
1        ...
10       ...
20       ...
50       ...
100      ...
```

---

# Experiment 06 — Numerical Gradient Check

**Status:** Planned

## Objective

Verify that our manually implemented gradients are correct.

Compare:

```text
Analytical Gradient
        vs
Numerical Gradient
```

Numerical approximation:

```text
f(x + ε) - f(x - ε)
-------------------
        2ε
```

---

## Success Criteria

The analytical and numerical gradients should be very close.

This experiment is particularly important because a small error in backpropagation can make training fail.

---

# Experiment 07 — Character Prediction

**Status:** Planned

## Objective

Move from numerical relationships to text.

Train a tiny model to predict the next character.

Example:

```text
Input     Target

h         e
he        l
hel       l
hell      o
hello     space
```

---

## New Concepts

* Text data
* Character vocabulary
* Character IDs
* Sequence data
* Next-character prediction

---

# Experiment 08 — Character Tokenizer

**Status:** Planned

## Objective

Convert text into numerical representations.

Example:

```text
"hello"
```

could become:

```text
[7, 4, 11, 11, 14]
```

and back:

```text
[7, 4, 11, 11, 14]
        ↓
"hello"
```

---

## New Concepts

* Vocabulary
* Token IDs
* Encode
* Decode
* Unknown tokens
* Special tokens

---

# Experiment 09 — Embeddings

**Status:** Planned

## Objective

Replace simple token IDs with learned vector representations.

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

---

## Questions

* Why isn't an integer token ID enough?
* What does an embedding represent?
* How are embeddings learned?
* Do related tokens develop similar representations?

---

# Experiment 10 — Self-Attention

**Status:** Planned

## Objective

Implement self-attention from scratch.

Workflow:

```text
Input embeddings
       ↓
Query
Key
Value
       ↓
Attention scores
       ↓
Weighted values
       ↓
Context representation
```

---

## Initial Scope

Start with:

```text
Single-head self-attention
```

Do not start with multi-head attention.

---

## Questions

* What does Query represent?
* What does Key represent?
* What does Value represent?
* How are attention scores calculated?
* Why does scaling matter?
* What information does attention capture?

---

# Experiment 11 — Multi-Head Attention

**Status:** Planned

## Objective

Extend single-head attention into multiple attention heads.

```text
Input
  ↓
 ┌───────┬───────┬───────┬───────┐
Head 1  Head 2  Head 3  Head 4
 └───────┴───────┴───────┴───────┘
              ↓
           Combine
```

---

## Questions

* Why use multiple heads?
* Do different heads learn different relationships?
* What happens when the number of heads changes?

---

# Experiment 12 — Transformer Block

**Status:** Planned

## Objective

Build the first complete Transformer block.

Architecture:

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

---

## New Concepts

* Residual connections
* Layer normalization
* Feed-forward networks
* Transformer blocks

---

# Experiment 13 — Tiny Transformer

**Status:** Planned

## Objective

Stack Transformer blocks and build a tiny language model.

Architecture:

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
Transformer Block
 ↓
Transformer Block
 ↓
Linear Output
 ↓
Logits
 ↓
Next Token
```

---

## Training Objective

Next-token prediction.

```text
Input:

"The pharmacy filled the"

Target:

"prescription"
```

---

# Experiment 14 — Tiny Language Model

**Status:** Planned

## Objective

Train a complete small language model on a controlled dataset.

The model should be capable of generating short sequences based on learned patterns.

---

## Metrics

Track:

```text
Number of parameters
Training dataset size
Training time
Epochs
Learning rate
Final loss
Validation loss
Generation examples
```

---

# Experiment 15 — Sampling

**Status:** Planned

## Objective

Understand how a language model converts probabilities into generated text.

Compare:

```text
Greedy decoding
Temperature
Top-K
Top-P
```

---

## Questions

* Why doesn't the model always select the highest-probability token?
* How does temperature affect generation?
* What happens when temperature is very low?
* What happens when temperature is very high?

---

# Experiment 16 — Overfitting

**Status:** Planned

## Objective

Deliberately make the model overfit a small dataset.

Observe:

```text
Training Loss ↓↓↓
Validation Loss ↑
```

---

## Questions

* What is overfitting?
* Why does it happen?
* How can it be detected?
* What techniques can reduce it?

---

# Experiment 17 — NumPy Comparison

**Status:** Planned

## Objective

Reimplement selected components using NumPy.

Compare:

```text
Pure Python
vs
NumPy
```

Measure:

```text
Execution time
Memory usage
Implementation complexity
```

---

# Experiment 18 — PyTorch Comparison

**Status:** Planned

## Objective

Implement the same basic model using PyTorch.

Compare:

```text
Our implementation
vs
PyTorch
```

The purpose is not simply performance comparison.

The purpose is to understand what PyTorch abstracts away.

---

# Experiment 19 — C++ Implementation

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

# Experiment 20 — Rust Implementation

**Status:** Future

## Objective

Implement a selected performance-critical component in Rust.

Explore:

* Ownership
* Borrowing
* Memory management
* Concurrency
* Performance
* Python/Rust interoperability

---

# Experiment 21 — GPU Exploration

**Status:** Future

## Objective

Understand why GPUs accelerate neural-network workloads.

Explore:

```text
CPU
 ↓
GPU
 ↓
Parallel computation
 ↓
Matrix operations
```

Later investigate CUDA and GPU kernels.

---

# Experiment Template

Every new experiment should use the following structure.

```markdown
# Experiment XX — Name

**Status:** Planned / In Progress / Completed

## Objective

What are we trying to learn?

## Hypothesis

What do we expect to happen?

## Implementation

What did we build?

## Configuration

- Dataset:
- Parameters:
- Learning rate:
- Epochs:
- Batch size:

## Results

What happened?

## Metrics

- Initial loss:
- Final loss:
- Training time:
- Parameter count:

## Observations

What did we notice?

## Problems

What failed or behaved unexpectedly?

## What We Learned

What did this experiment teach us?

## Next Step

What should we investigate next?
```

---

# Experiment Rules

## Rule 1 — Record failures

A failed experiment is still valuable.

Document:

```text
What failed
Why it failed
How it was fixed
What we learned
```

---

## Rule 2 — Change one important variable at a time

When investigating behavior, avoid changing everything simultaneously.

For example:

```text
learning rate
```

should be tested independently before changing:

```text
architecture
dataset
batch size
optimizer
```

This makes the results easier to understand.

---

## Rule 3 — Keep experiments reproducible

Record:

```text
Random seed
Dataset
Configuration
Code version
```

Whenever practical.

---

## Rule 4 — Measure before optimizing

Do not assume something is slow.

Measure it.

Later comparisons should include actual timing and memory measurements.

---

## Rule 5 — Understand before using frameworks

Before introducing a framework abstraction, first understand the underlying concept.

Examples:

```text
Manual gradient
      ↓
Automatic differentiation

Manual matrix operations
      ↓
NumPy

Manual neural network
      ↓
PyTorch

Manual attention
      ↓
Transformer implementation
```

---

# Current Experiment

## Experiment 01 — Linear Learning

Our immediate target:

```text
Random weight
      ↓
Forward pass
      ↓
Loss
      ↓
Gradient
      ↓
Backpropagation
      ↓
Gradient descent
      ↓
Weight ≈ 3
```

Dataset:

```text
1 → 3
2 → 6
3 → 9
4 → 12
5 → 15
```

**Next action:** implement the first neuron and training loop.

---

# Long-Term Goal

Start with:

```text
y = 3x
```

and progressively build toward:

```text
Tiny Transformer
       ↓
Tiny Language Model
```

The important outcome is not the size of the final model.

The important outcome is:

> **Understanding how the model learns.**
