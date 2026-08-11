# LLM From Scratch — Experiments

This document records the practical experiments performed during the development of the project.

The purpose is not only to record successful results, but also to document:

- What we tried
- Why we tried it
- What happened
- What failed
- What we learned
- What we will change next

The experiments progressively move from simple mathematical models toward a tiny language model.

---

# Experiment 01 — Linear Learning

**Status:** Completed

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

The model starts with an incorrect weight rather than being given the value `3`.

---

## Model

```text
prediction = weight × input
```

Initial weight:

```text
weight = 0.5
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

Core files:

```text
src/llm_from_scratch/core/
├── neuron.py
├── loss.py
├── optimizer.py
└── training.py
```

Tests:

```text
tests/
├── test_neuron.py
├── test_loss.py
├── test_optimizer.py
└── test_training.py
```

Experiment:

```text
experiments/
└── 01_linear_learning.py
```

---

## Parameters

```text
Initial weight:    0.5
Learning rate:     0.01
Epochs:            100
Training examples: 5
```

---

## Results

```text
Initial weight: 0.5000
Final weight:   3.000000
Final loss:     0.000000
```

Training output showed:

```text
Epoch   1 | Loss: 34.026687 | Weight: 2.371585
Epoch  10 | Loss: 0.000000 | Weight: 2.999997
Epoch 100 | Loss: 0.000000 | Weight: 3.000000
```

---

## Observations

The model successfully discovered the relationship:

```text
y = 3x
```

without being explicitly given:

```text
weight = 3
```

The weight was updated through repeated:

```text
Prediction
    ↓
Loss
    ↓
Gradient
    ↓
Weight Update
```

The loss reached zero because the relationship can be represented exactly by our model.

---

## What We Learned

- A neuron can make a prediction using a parameter.
- Loss measures how wrong the prediction is.
- A gradient tells us how the parameter should change.
- Gradient descent updates the parameter.
- Repeating this process is training.
- A model can discover parameters instead of being directly given the correct values.

---

# Experiment 02 — Linear Model Limitation

**Status:** Completed

## Objective

Observe what happens when a simple linear model is given training data that cannot be represented by a single weight.

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

The first five examples follow:

```text
y = 3x
```

After `x = 5`, the target becomes constant:

```text
y = 15
```

---

## Model

The model remains:

```text
prediction = weight × input
```

It still has only one trainable parameter:

```text
weight
```

---

## Experiment

The model starts with:

```text
weight = 0.5
```

and attempts to minimize the total loss across all training examples.

---

## Results

```text
Initial weight: 0.5000
Final weight:   1.585380
Final loss:     7.692325
```

Training output:

```text
Epoch   1 | Loss: 20.222011 | Weight: 1.585115
Epoch  10 | Loss:  7.692325 | Weight: 1.585380
Epoch 100 | Loss:  7.692325 | Weight: 1.585380
```

---

## Observation

The loss decreased:

```text
20.222011
     ↓
7.692325
```

but never reached zero.

The model found a compromise rather than perfectly fitting the data.

This happens because no single value of `weight` can satisfy all examples.

For example:

```text
1 → 3    requires weight = 3
2 → 6    requires weight = 3
3 → 9    requires weight = 3
4 → 12   requires weight = 3
5 → 15   requires weight = 3
```

But:

```text
6 → 15   requires weight = 2.5
7 → 15   requires weight ≈ 2.14
8 → 15   requires weight = 1.875
9 → 15   requires weight ≈ 1.67
```

There is no single weight that satisfies both behaviors.

---

## What We Learned

A model can only learn relationships that its architecture is capable of representing.

Our model:

```text
prediction = weight × input
```

can represent a straight-line relationship through the origin.

It cannot represent:

```text
linear growth
+
constant value
```

at the same time.

The optimizer is not failing.

The model is simply **too simple for the problem**.

This is an important machine-learning concept:

> When the model cannot represent the underlying relationship, training may find the best available approximation instead of a perfect solution.

---

## Next Direction

This experiment motivates the need for more expressive models.

We will progressively introduce:

```text
Bias
  ↓
Multiple Neurons
  ↓
Activation Functions
  ↓
Multiple Layers
```

---

# Experiment 03 — Add Bias

**Status:** In Progress

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

## Training Data

```text
x    y
---------
1    5
2    8
3    11
4    14
5    17
```

---

## Initial Parameters

The model starts with incorrect parameters:

```text
weight = 0.5
bias   = 0.0
```

The `bias` parameter defaults to `0.0` so the neuron remains backward compatible with the original linear model.

---

## New Concepts

- Bias
- Multiple trainable parameters
- Weight gradient
- Bias gradient
- Multiple parameter updates

---

## Forward Pass

The neuron now calculates:

```text
prediction = (weight × input) + bias
```

Example:

```text
weight = 3
bias   = 2
input  = 4

prediction = (3 × 4) + 2
           = 14
```

---

## Backward Pass

The neuron calculates separate gradients for both trainable parameters:

```text
Weight Gradient
       ↓
Weight Update

Bias Gradient
       ↓
Bias Update
```

The gradients are:

```text
weight_gradient = output_gradient × input

bias_gradient = output_gradient
```

---

## Training Workflow

```text
Input
  ↓
Forward Pass
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
       ├── Weight
       └── Bias
          ↓
Updated Parameters
  ↓
Repeat
```

---

## Implementation

Core implementation:

```text
src/llm_from_scratch/core/
├── neuron.py
├── loss.py
├── optimizer.py
└── training.py
```

The neuron now supports:

```text
weight
bias
weight gradient
bias gradient
```

The training step updates both `weight` and `bias`.

---

## Tests

The neuron tests verify:

```text
Forward pass with bias
Weight gradient
Bias gradient
```

The training tests verify:

```text
Weight is updated
Bias is updated
```

Current test suite:

```text
10 tests passed
```

---

## Experiment

```text
experiments/
└── 03_bias_learning.py
```

The experiment will train the neuron on:

```text
y = 3x + 2
```

and measure whether it can discover:

```text
weight ≈ 3
bias   ≈ 2
```

---

## Results

**Pending experiment execution.**

Record after training:

```text
Initial weight:
Initial bias:

Final weight:
Final bias:

Initial loss:
Final loss:

Epochs:
Learning rate:
Training time:
```

---

## Questions

- Why do we need bias?
- How does bias change what a neuron can represent?
- Can both parameters converge?
- How does the gradient differ for weight and bias?
- Does adding bias allow the model to represent relationships that the original neuron could not?

---

## What We Learned

The neuron now has more than one trainable parameter.

Instead of learning only:

```text
weight
```

it can learn:

```text
weight
bias
```

The optimizer itself does not need to know what a parameter represents. It simply receives:

```text
parameter
gradient
learning rate
```

and returns an updated parameter.

This allows the same optimizer to update both weight and bias.

---

# Experiment 04 — Multiple Neurons

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

- Layers
- Multiple weights
- Multiple biases
- Vectors
- Matrix operations

---

## Questions

- How do multiple neurons cooperate?
- What does each neuron learn?
- Does adding neurons improve model capacity?

---

# Experiment 05 — Activation Functions

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

- Why do neural networks need non-linear activation?
- What happens when all layers are linear?
- How does ReLU change the output?
- What happens to gradients?

---

# Experiment 06 — Training Loop

**Status:** Completed

## Objective

Build a reusable training step and use it inside an experiment-level training loop.

The reusable training step performs:

```text
Forward
  ↓
Loss
  ↓
Loss Gradient
  ↓
Neuron Gradient
  ↓
Optimizer
  ↓
Updated Parameters
```

The experiment-level loop repeats this process over:

```text
Epochs
    ↓
Training Examples
```

---

## Implementation

The training workflow is implemented in:

```text
src/llm_from_scratch/core/training.py
```

The trainer coordinates:

```text
Neuron
Loss
Optimizer
```

without implementing their internal mathematics.

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

# Experiment 07 — Numerical Gradient Check

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

# Experiment 08 — Character Prediction

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

- Text data
- Character vocabulary
- Character IDs
- Sequence data
- Next-character prediction

---

# Experiment 09 — Character Tokenizer

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

- Why isn't an integer token ID enough?
- What does an embedding represent?
- How are embeddings learned?
- Do related tokens develop similar representations?

---

# Experiment 11 — Self-Attention

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

- What does Query represent?
- What does Key represent?
- What does Value represent?
- How are attention scores calculated?
- Why does scaling matter?
- What information does attention capture?

---

# Experiment 12 — Multi-Head Attention

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

- Why use multiple heads?
- Do different heads learn different relationships?
- What happens when the number of heads changes?

---

# Experiment 13 — Transformer Block

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

- Residual connections
- Layer normalization
- Feed-forward networks
- Transformer blocks

---

# Experiment 14 — Tiny Transformer

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

# Experiment 15 — Tiny Language Model

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

# Experiment 16 — Sampling

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

- Why doesn't the model always select the highest-probability token?
- How does temperature affect generation?
- What happens when temperature is very low?
- What happens when temperature is very high?

---

# Experiment 17 — Overfitting

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

- What is overfitting?
- Why does it happen?
- How can it be detected?
- What techniques can reduce it?

---

# Experiment 18 — NumPy Comparison

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

# Experiment 19 — PyTorch Comparison

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

# Current Progress

Completed:

```text
01 — Linear Learning             ✅
02 — Linear Model Limitation     ✅
06 — Training Loop               ✅
```

In Progress:

```text
03 — Add Bias                    🔄
```

The current implementation has successfully demonstrated:

```text
Neuron
  ↓
Forward Pass
  ↓
Loss
  ↓
Gradient
  ↓
Backpropagation
  ↓
Gradient Descent
  ↓
Parameter Update
  ↓
Training Loop
```

The neuron now supports:

```text
Weight
Bias
Weight Gradient
Bias Gradient
```

The training step updates both trainable parameters.

---

# Immediate Next Step

Complete:

```text
03 — Add Bias
```

Create:

```text
experiments/03_bias_learning.py
```

Train the model on:

```text
y = 3x + 2
```

and verify whether it can discover:

```text
weight ≈ 3
bias   ≈ 2
```

After the experiment is executed, record the actual training results in this document.

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
