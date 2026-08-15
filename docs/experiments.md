# LLM From Scratch --- Experiments

This document records the experiments used to understand and build the
project from simple mathematical models toward a tiny language model.

Each experiment focuses on:

- What we are trying to learn
- What we implemented
- What happened
- Major learning
- Important exceptions / edge cases
- Short summary
- What comes next

---

# Experiment 01 --- Linear Learning

**Status:** Completed

## Objective

Teach the smallest possible model to discover:

```text
y = 3 × x
```

## Model

```text
prediction = weight × input
```

## Result

The model started with:

```text
weight = 0.5
```

and learned approximately:

```text
weight = 3.0
loss   ≈ 0
```

## Major Learning

A model can learn a parameter from examples rather than being explicitly
given the correct parameter.

The fundamental learning loop is:

```text
Prediction
    ↓
Loss
    ↓
Gradient
    ↓
Parameter Update
    ↓
Repeat
```

## Important Exception

A model can only learn what its architecture can represent. A simple
linear model is limited to linear relationships.

## Short Summary

> The first experiment proved that gradient-based learning works.

---

# Experiment 02 --- Linear Model Limitation

**Status:** Completed

## Objective

Demonstrate what happens when the model is too simple for the data.

The data begins with a linear relationship and then becomes constant:

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

The loss decreased but did not reach zero.

## Major Learning

The optimizer was not necessarily failing.

The model itself could not represent both:

```text
y = 3 × x
```

and:

```text
y = 15
```

with a single weight.

This introduced an important principle:

> Model architecture determines what relationships the model can
> represent.

## Important Exception

More training does not solve every problem. If the model is incapable of
representing the target relationship, additional epochs cannot fix the
architectural limitation.

## Short Summary

> A better optimizer cannot compensate for an insufficient model
> architecture.

---

# Experiment 03 --- Bias

**Status:** Completed

## Objective

Extend the neuron from:

```text
y = weight × input
```

to:

```text
y = weight × input + bias
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

## Major Learning

A neuron does not need to force its prediction through zero.

The bias allows the model to shift the output.

## Important Exception

Bias is useful for representational flexibility, but it does not
introduce non-linearity. A network containing only linear
transformations is still fundamentally linear.

## Short Summary

> Weight controls the relationship with the input; bias shifts the
> output.

---

# Experiment 04 --- Multiple Neurons

**Status:** Completed

## Objective

Move from:

```text
Input → Neuron → Output
```

to:

```text
Input
  ↓
Multiple Neurons
  ↓
Multiple Outputs
```

Example:

```text
              ┌── Neuron 1 → Output 1
Input ────────┼── Neuron 2 → Output 2
              └── Neuron 3 → Output 3
```

## Result

The model successfully learned separate relationships:

```text
Neuron 1 → y = 3 × x
Neuron 2 → y = 5 × x
Neuron 3 → y = 7 × x
```

After extended training:

```text
weights ≈ [3.0, 5.0, 7.0]
```

with loss approaching zero.

## Major Learning

Multiple neurons can learn different relationships from the same input.

This led to the introduction of the `Layer` abstraction.

## Important Exception

Multiple neurons alone do not make a network deep. They are still
operating at the same layer.

## Short Summary

> A layer is a coordinated collection of independent neurons.

---

# Experiment 05 --- Activation Functions / ReLU

**Status:** Completed

## Objective

Introduce non-linearity using ReLU.

ReLU stands for **Rectified Linear Unit**:

```text
ReLU(x) = max(0, x)
```

Forward:

```text
positive → unchanged
negative → 0
zero     → 0
```

Backward:

```text
raw input > 0 → gradient passes through
raw input ≤ 0 → gradient becomes 0
```

## Experiment

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

The model converged approximately to:

```text
weight ≈ 2
bias   ≈ 0
loss   ≈ 0
```

## Major Learning

Activation functions introduce non-linearity.

Without activation:

```text
Layer → Linear → Layer → Linear
```

still behaves as a linear transformation.

With activation:

```text
Layer → Linear → ReLU → Layer → Linear → ReLU
```

the network can represent non-linear relationships.

## Important Exception

ReLU has a zero-gradient region:

```text
x ≤ 0 → gradient = 0
```

A neuron that remains in that region may stop receiving useful gradient
updates. This is one reason activation choice matters.

## Short Summary

> ReLU makes stacked neural layers capable of learning non-linear
> relationships and controls gradient flow.

---

# Experiment 06 --- Multi-Input Neuron

**Status:** Completed

## Objective

Generalize the neuron from one input to an input vector.

Instead of:

```text
weight × input + bias
```

the neuron now calculates:

```text
(weight1 × input1)
+ (weight2 × input2)
+ ...
+ bias
```

Conceptually:

```text
Input Vector
    ↓
Weighted Sum
    ↓
Bias
    ↓
Activation
    ↓
Output
```

## Implementation

The neuron now stores:

```text
weights: list[float]
bias
```

and calculates a gradient for every weight.

## Major Learning

A neuron can process multiple features simultaneously.

This is the foundation required for real dense neural-network layers.

## Important Exception

The number of inputs must match the number of weights.

A mismatch is rejected rather than silently producing an incorrect
calculation.

## Short Summary

> One neuron can learn a weighted combination of multiple input
> features.

---

# Experiment 07 --- Layer Gradient Aggregation

**Status:** Completed

## Objective

Generalize a layer so multiple neurons can process the same input vector
and return a single gradient vector for the previous layer.

```text
Input Vector
      ↓
 ┌────┼────┐
 ↓    ↓    ↓
 N1   N2   N3
 ↓    ↓    ↓
G1   G2   G3
 └────┼────┘
      ↓
Aggregated Input Gradient
```

## Major Learning

During backward propagation, each neuron produces a gradient with
respect to every input.

The layer must aggregate those contributions.

For an input:

```text
x1
```

the layer receives contributions such as:

```text
gradient_from_neuron_1
+
gradient_from_neuron_2
+
gradient_from_neuron_3
```

## Important Exception

The returned gradient vector must match the input-vector dimensions.

This aggregation is essential when multiple neurons depend on the same
input.

## Short Summary

> A layer combines the gradient contributions from all of its neurons.

---

# Experiment 08 --- Network / Multiple Layers

**Status:** Completed

## Objective

Move from one layer to a sequence of layers.

```text
Input
  ↓
Layer 1
  ↓
Layer 2
  ↓
Output
```

## Network

The new `Network` abstraction coordinates layers.

Forward propagation:

```text
Input
  ↓
Layer 1
  ↓
Layer 2
  ↓
Output
```

Backward propagation:

```text
Output Gradient
      ↓
Layer 2
      ↓
Layer 1
      ↓
Input Gradient
```

## Verified Experiment

A two-layer network was tested with:

```text
Input:
[2.0, 3.0]
```

and produced:

```text
Output:
[4.48, 3.11]
```

A backward pass using:

```text
Output gradients:
[1.0, -2.0]
```

produced:

```text
Input gradients:
[-0.18, -0.16]
```

The calculations were verified manually.

## Major Learning

Backpropagation through multiple layers is simply the reverse flow of
the forward computation.

The output of one layer becomes the input to the next, while gradients
travel in the opposite direction.

## Important Exception

Backward propagation must process layers in reverse order.

```text
Forward:
Layer 1 → Layer 2

Backward:
Layer 2 → Layer 1
```

Using the wrong order would produce incorrect gradients.

## Short Summary

> A network is a sequence of layers, and backpropagation traverses that
> sequence in reverse.

---

# Experiment 09 --- Network-Based Training

**Status:** Completed

## Objective

Update the Trainer so it can train an entire `Network` rather than a
single `Layer`.

## Training Flow

```text
Network.forward()
      ↓
Prediction
      ↓
Loss.forward()
      ↓
Loss.backward()
      ↓
Network.backward()
      ↓
Update every neuron
      ↓
Update every weight + bias
```

The Trainer now coordinates the complete network without implementing
the mathematical details of individual neurons or layers.

## Verification

The training tests verify:

- Loss reduction
- Weight updates
- Bias updates
- Multiple neuron updates
- Multiple-layer parameter updates

The complete test suite currently contains:

```text
28 passed
```

## Major Learning

The architecture is now separated into clear responsibilities:

```text
Neuron
  ↓
Mathematical unit

Layer
  ↓
Collection of neurons

Network
  ↓
Collection of layers

Loss
  ↓
Measures error

Optimizer
  ↓
Updates parameters

Trainer
  ↓
Coordinates training
```

## Important Exception

The Trainer should not contain neuron or layer mathematics.

It coordinates the components; each component remains responsible for
its own behavior.

## Short Summary

> We now have a reusable, tested training pipeline capable of training a
> multi-layer network.

---

# Experiment 10 --- Multi-Layer Training Experiment

**Status:** In Progress

## Objective

Train a real network containing multiple layers and observe both layers
learning from the same loss.

Target architecture:

```text
Input
  ↓
Layer 1
  ↓
ReLU
  ↓
Layer 2
  ↓
ReLU
  ↓
Output
  ↓
Loss
```

Training must demonstrate:

```text
Loss
  ↓
Layer 2 gradients
  ↓
Layer 1 gradients
  ↓
Update Layer 2
  ↓
Update Layer 1
```

## Current Progress

The forward and backward behavior of a two-layer network has already
been verified.

The next step is to run a complete training experiment and observe:

```text
Initial Loss
      ↓
Training
      ↓
Lower Loss
      ↓
Updated Layer 1 parameters
      ↓
Updated Layer 2 parameters
```

## Major Learning

This experiment will demonstrate the central idea behind deep learning:

> An error at the final output can influence parameters throughout
> earlier layers through backpropagation.

## Important Exception

A multi-layer architecture does not automatically guarantee better
learning.

Performance still depends on:

- Model architecture
- Activation functions
- Initialization
- Learning rate
- Training data
- Number of epochs
- Optimization behavior

## Short Summary

> The architecture for multi-layer learning is complete; the remaining
> task is to demonstrate end-to-end learning through multiple layers.

---

# Future Experiments

## Experiment 11 --- Numerical Gradient Check

**Status:** Planned

Verify analytical gradients against numerical gradients:

```text
f(x + ε) - f(x - ε)
-------------------
        2ε
```

### Major Learning

Gradient checking gives us an independent way to validate
backpropagation.

### Important Exception

Numerical gradient checking is primarily a debugging/verification tool.
It is not how we will train the final model because it is
computationally expensive.

### Short Summary

> Use numerical gradients to verify our analytical gradients.

---

# Experiment 12 --- Character Prediction

**Status:** Planned

Move from numerical relationships to text.

Example:

```text
Input → Target

h     → e
he    → l
hel   → l
hell  → o
```

### Major Learning

The same prediction → loss → gradient → update loop can be applied to
language data.

### Important Exception

Text must first be represented numerically before our neural network can
process it.

### Short Summary

> The next major transition is from numeric data to language data.

---

# Experiment 13 --- Character Tokenizer

**Status:** Planned

Convert text into numerical representations:

```text
"hello"
   ↓
[7, 4, 11, 11, 14]
```

and decode back:

```text
[7, 4, 11, 11, 14]
   ↓
"hello"
```

### Major Learning

Tokenization converts language into discrete numerical units that a
model can process.

### Important Exception

Token IDs themselves are categorical identifiers. The numeric distance
between IDs does not represent semantic similarity.

### Short Summary

> Tokenization creates the numerical vocabulary used by the model.

---

# Experiment 14 --- Embeddings

**Status:** Planned

Convert token IDs into learned vectors:

```text
Token ID
   ↓
Embedding Matrix
   ↓
Vector
```

### Major Learning

Embeddings give the model a learnable continuous representation of
tokens.

### Important Exception

An embedding is not inherently a dictionary definition. Its meaning
emerges from how it is learned from context.

### Short Summary

> Embeddings turn discrete tokens into learnable vectors.

---

# Experiment 15 --- Self-Attention

**Status:** Planned

Implement single-head self-attention:

```text
Embeddings
    ↓
Query / Key / Value
    ↓
Attention Scores
    ↓
Weighted Values
    ↓
Context Representation
```

### Major Learning

Attention allows the model to dynamically determine which parts of the
input are important to the current representation.

### Important Exception

Attention is not simply a lookup table. The attention weights are
calculated from the current representations.

### Short Summary

> Attention allows information from different positions to interact.

---

# Experiment 16 --- Multi-Head Attention

**Status:** Planned

Extend self-attention to multiple heads.

```text
Input
  ↓
Head 1
Head 2
Head 3
...
  ↓
Combine
```

### Major Learning

Different attention heads can learn different relationships or patterns.

### Important Exception

More heads do not automatically mean a better model. The number of heads
must work with the model dimension and architecture.

### Short Summary

> Multi-head attention lets the model learn several attention patterns
> in parallel.

---

# Experiment 17 --- Transformer Block

**Status:** Planned

Build:

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

### Major Learning

A Transformer block combines attention with residual connections and
feed-forward computation into a reusable building block.

### Important Exception

Removing or changing components such as residual connections or
normalization can significantly affect training behavior.

### Short Summary

> A Transformer block combines several core neural-network mechanisms
> into one reusable unit.

---

# Experiment 18 --- Tiny Transformer

**Status:** Planned

Stack Transformer blocks:

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

### Major Learning

This connects the individual concepts into a language-model
architecture.

### Important Exception

A tiny Transformer is educational. Its size and dataset will be far
below production LLM scale.

### Short Summary

> The Transformer combines our earlier concepts into a language model
> architecture.

---

# Experiment 19 --- Tiny Language Model

**Status:** Planned

Train a small language model on a controlled dataset.

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

### Major Learning

Observe how a complete language model behaves during training.

### Important Exception

Low training loss does not automatically mean good language generation
or generalization.

### Short Summary

> A complete tiny LLM lets us connect the theory to actual text
> generation.

---

# Experiment 20 --- Sampling

**Status:** Planned

Compare:

```text
Greedy decoding
Temperature
Top-K
Top-P
```

### Major Learning

Generation is a probability-selection process rather than simply
selecting the highest-scoring token every time.

### Important Exception

Sampling parameters affect creativity, diversity, and consistency; there
is no universally best setting.

### Short Summary

> Sampling controls how model probabilities become generated text.

---

# Experiment 21 --- Overfitting

**Status:** Planned

Deliberately overfit a small dataset and observe:

```text
Training Loss ↓↓↓
Validation Loss ↑
```

### Major Learning

A model can memorize training examples without learning patterns that
generalize.

### Important Exception

Low training loss alone is not sufficient evidence that the model has
learned a useful representation.

### Short Summary

> Training performance and generalization are different things.

---

# Experiment 22 --- NumPy Comparison

**Status:** Planned

Compare selected pure-Python implementations with NumPy.

Measure:

```text
Execution time
Memory usage
Implementation complexity
```

### Major Learning

Understand why numerical libraries are useful for vectorized
computation.

### Important Exception

Optimization should follow measurement rather than assumptions.

### Short Summary

> NumPy shows how optimized numerical operations change the
> implementation and performance.

---

# Experiment 23 --- PyTorch Comparison

**Status:** Planned

Implement the same basic model using PyTorch.

### Major Learning

Understand what tensors, automatic differentiation, and optimized
training frameworks provide.

### Important Exception

The framework should be introduced after understanding the underlying
mathematics, not used as a replacement for understanding it.

### Short Summary

> PyTorch should make more sense after we have already built the
> fundamentals ourselves.

---

# Experiment 24 --- C++ Implementation

**Status:** Future

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

### Major Learning

Understand lower-level performance and memory behavior.

### Important Exception

C++ is not automatically faster for every workload; the implementation
and workload matter.

### Short Summary

> Reimplement a selected computational core to understand systems-level
> performance.

---

# Experiment 25 --- Rust Implementation

**Status:** Future

Implement a selected performance-critical component in Rust.

Explore:

- Ownership
- Borrowing
- Memory management
- Concurrency
- Performance
- Python/Rust interoperability

### Major Learning

Understand how Rust approaches safe, high-performance systems
programming.

### Important Exception

The goal is learning and comparison, not replacing Python everywhere.

### Short Summary

> Use Rust to explore safe systems-level implementations of selected
> model components.

---

# Experiment 26 --- GPU Exploration

**Status:** Future

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

### Major Learning

Understand why large neural networks depend heavily on parallel
numerical computation.

### Important Exception

Moving code to a GPU does not automatically make every operation faster;
workload size and data-transfer overhead matter.

### Short Summary

> GPU exploration connects our mathematical implementation to modern ML
> hardware.

---

# Current Progress

Completed:

```text
01 — Linear Learning              ✅
02 — Linear Model Limitation      ✅
03 — Bias                         ✅
04 — Multiple Neurons             ✅
05 — Activation / ReLU            ✅
06 — Multi-Input Neuron           ✅
07 — Layer Gradient Aggregation   ✅
08 — Network / Multiple Layers    ✅
09 — Network-Based Training       ✅
```

In progress:

```text
10 — Multi-Layer Training         🔄
```

Planned:

```text
11 — Numerical Gradient Check
12 — Character Prediction
13 — Character Tokenizer
14 — Embeddings
15 — Self-Attention
16 — Multi-Head Attention
17 — Transformer Block
18 — Tiny Transformer
19 — Tiny Language Model
20 — Sampling
21 — Overfitting
22 — NumPy Comparison
23 — PyTorch Comparison
```

Future systems track:

```text
24 — C++ Implementation
25 — Rust Implementation
26 — GPU Exploration
```

---

# Current Architecture

The project has evolved into:

```text
Input Vector
      ↓
Network
      ↓
Layer 1
      ↓
Neuron(s)
      ↓
Weighted Sum + Bias
      ↓
Activation
      ↓
Layer Output
      ↓
Layer 2
      ↓
...
      ↓
Prediction
      ↓
Loss
      ↓
Backpropagation
      ↓
Gradient Descent
      ↓
Updated Parameters
```

Component responsibilities:

```text
Neuron
  → mathematical unit

Activation
  → non-linearity and gradient gating

Layer
  → collection of neurons and gradient aggregation

Network
  → sequence of layers

Loss
  → measures prediction error

Optimizer
  → updates parameters

Trainer
  → coordinates the learning cycle
```

---

# Major Milestone

The project has moved beyond a single-neuron demonstration.

We now have a **tested, multi-input, multi-neuron, multi-layer training
architecture** implemented from first principles.

The complete test suite currently contains:

```text
28 passed
```

The most important architectural transition was:

```text
Single Neuron
      ↓
Layer
      ↓
Network
      ↓
Network-Based Trainer
```

This gives us the foundation required for the next major transition:

```text
Numeric Learning
      ↓
Language Learning
      ↓
Tokenizer
      ↓
Embeddings
      ↓
Attention
      ↓
Transformer
      ↓
Tiny LLM
```

---

# Exceptions and Important Lessons So Far

Several important lessons have already emerged:

1.  **More epochs do not fix an architecture that cannot represent the
    target relationship.**

2.  **More neurons do not automatically make a model deep.** Depth comes
    from stacking layers.

3.  **Linear layers without non-linear activation remain effectively
    linear.**

4.  **ReLU can block gradients when its raw input is not positive.**

5.  **Input and weight dimensions must match.**

6.  **Backward propagation must process layers in reverse order.**

7.  **The Trainer should coordinate components rather than duplicate
    their mathematical logic.**

8.  **Low loss is not always sufficient evidence of useful learning.**
    Generalization will become important later.

---

# Short Summary

So far, the project has demonstrated the complete fundamental learning
mechanism:

```text
Input
  ↓
Prediction
  ↓
Loss
  ↓
Gradient
  ↓
Backpropagation
  ↓
Parameter Update
  ↓
Better Prediction
```

We then expanded that mechanism:

```text
Neuron
  ↓
Multiple Inputs
  ↓
Layer
  ↓
Multiple Layers
  ↓
Network
  ↓
Network-Based Training
```

The **next immediate goal** is to complete Experiment 10 by training the
two-layer network end-to-end and observing both layers learn from the
final loss.

After that, we will make the major conceptual transition from numerical
learning to language:

```text
Character Prediction
      ↓
Tokenization
      ↓
Embeddings
      ↓
Attention
      ↓
Transformer
      ↓
Tiny LLM
```

> **The objective is not simply to build a tiny LLM. The objective is to
> understand every major mechanism that makes the LLM learn.**
