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

The complete test suite is passing after the gradient-checking
milestone.

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

**Status:** Completed

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

## Result

A two-layer network successfully learned the relationship:

```text
y = 2 × x + 1
```

using:

```text
Layer 1 → 2 neurons → ReLU
Layer 2 → 1 neuron  → Linear
```

Training loss converged:

```text
Epoch 100  | Loss: 0.080190
Epoch 200  | Loss: 0.002423
Epoch 300  | Loss: 0.000070
Epoch 400  | Loss: 0.000002
Epoch 500  | Loss: 0.000000
```

The final predictions matched all training targets:

```text
Input   Target   Prediction
---------------------------
-2.0    -3.0     -3.0000
-1.0    -1.0     -1.0000
 0.0     1.0      1.0000
 1.0     3.0      3.0000
 2.0     5.0      5.0000
 3.0     7.0      7.0000
```

Both layers learned new parameter values during training.

## Major Learning

This experiment demonstrated end-to-end multi-layer learning.

The complete flow is:

```text
Input
  ↓
Layer 1
  ↓
ReLU
  ↓
Layer 2
  ↓
Linear Output
  ↓
Loss
  ↓
Network.backward()
  ↓
Layer 2 gradients
  ↓
Layer 1 gradients
  ↓
Gradient Descent
  ↓
Updated parameters
```

The hidden layer learned an internal representation without being given
explicit instructions about what each hidden neuron should represent.

The learned parameters do not need to match a manually derived solution.
Different parameter combinations can represent the same target
relationship.

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

> A complete two-layer network successfully learned a target relationship
> end-to-end, proving that gradients can flow through multiple layers and
> update parameters throughout the network.

---

# Experiment 11 --- Numerical Gradient Check

**Status:** Completed

## Objective

Verify analytical gradients against numerical gradients:

```text
f(x + ε) - f(x - ε)
-------------------
        2ε
```

## Result

Gradient checking was applied across the neural-network foundation:

```text
Neuron → Layer → Network
```

The analytical gradients matched the numerical gradients within the
configured tolerance.

## Major Learning

Gradient checking gives us an independent way to verify that our
backpropagation implementation is mathematically correct.

## Important Exception

Numerical gradient checking is a verification/debugging technique. It is
not practical as the training mechanism because it requires many extra
forward passes.

## Short Summary

> Numerical gradient checking independently validates analytical
> backpropagation gradients.

---

# Experiment 12 --- Character Language Model Foundation

**Status:** Completed

## Objective

Move from numerical relationships to character-level language modeling.

The first language-model pipeline introduced:

```text
Text
  ↓
Vocabulary
  ↓
Character IDs
  ↓
Next-Character Dataset
  ↓
Network
  ↓
Logits
  ↓
Softmax + Cross Entropy
  ↓
Backpropagation
  ↓
Gradient Descent
```

## Result

The first character prediction experiment successfully learned the
unambiguous mappings:

```text
h → e
e → l
```

but could not uniquely solve:

```text
l → l
l → o
```

because the same input character had two different targets.

## Major Learning

A model cannot produce different predictions for identical input without
additional information.

This exposed the need for context.

## Important Exception

The limitation was not caused by a broken optimizer or failed
backpropagation. The input representation itself did not contain enough
information.

## Short Summary

> The first character model proved that the neural-network training
> pipeline can be applied to language, while exposing the need for
> contextual information.

---

# Experiment 13 --- Context Prediction

**Status:** Completed

## Objective

Provide multiple previous characters as context for next-character
prediction.

For:

```text
hello
```

with:

```text
context_size = 2
```

the dataset becomes:

```text
he → l
el → l
ll → o
```

## Result

The model learned all three examples with very high confidence.

Final loss:

```text
0.009361
```

Predictions:

```text
Input   Target   Prediction
---------------------------
he      l        l
el      l        l
ll      o        o
```

Final probabilities were approximately:

```text
he → l    99.8898%
el → l    98.5324%
ll → o    98.8142%
```

## Major Learning

Adding context resolves the ambiguity that existed in the one-character
model.

Instead of:

```text
l → ?
```

the model can distinguish:

```text
el → l
ll → o
```

## Important Exception

The model is still receiving raw token IDs such as:

```text
h → 0
e → 1
l → 2
o → 3
```

Those IDs are categorical identifiers, not meaningful continuous
representations.

## Short Summary

> Context provides information that allows the model to distinguish
> different occurrences of the same character.

---

# Experiment 14 --- Embedding

**Status:** Completed

## Objective

Replace raw token IDs with learnable vector representations.

The embedding matrix maps:

```text
Token ID
   ↓
Learned Vector
```

For this experiment:

```text
Vocabulary size = 4
Embedding size  = 3
```

## Implementation

The `Embedding` component provides:

```text
forward()
backward()
```

Forward performs a row lookup:

```text
token ID
   ↓
embedding matrix row
   ↓
vector
```

Backward accumulates gradients into the embedding rows used by the
current input.

Repeated token IDs accumulate their gradient contributions into the same
embedding row.

## Verification

Embedding tests verify:

- Forward vector lookup
- Correct row selection
- Backward gradient accumulation
- Repeated-token gradient accumulation
- Embedding weight gradient checking

The embedding gradient was independently compared with a numerical
gradient and passed within the configured tolerance.

## Embedded Sequence

`EmbeddedSequence` was added as a small adapter:

```text
Token IDs
   ↓
Embedding
   ↓
Vectors
   ↓
Flatten
   ↓
Existing Network
```

Its backward path reshapes the network input gradients and passes them
back into the embedding.

## Language Trainer Integration

`LanguageTrainer` was extended so that an optional embedding can
participate in the same training step.

The complete flow is now:

```text
Token IDs
   ↓
Embedding
   ↓
Flattened Vectors
   ↓
Network
   ↓
Logits
   ↓
SoftmaxCrossEntropy
   ↓
Loss
   ↓
Network.backward()
   ↓
Embedding.backward()
   ↓
Update Network Parameters
   ↓
Update Embedding Parameters
```

A focused integration test verifies that one training step changes both
network and embedding parameters.

## Embedded Context Experiment

The embedded context model was trained on:

```text
he → l
el → l
ll → o
```

Training loss improved from:

```text
0.010942
```

at epoch 100 to:

```text
0.000591
```

at epoch 1000.

Final predictions:

```text
Input   Target   Prediction
---------------------------
he      l        l
el      l        l
ll      o        o
```

Final probabilities were approximately:

```text
he → l    99.9388%
el → l    99.9493%
ll → o    99.9347%
```

## Embedding Learning

The embedding vectors changed during training.

For example:

```text
Initial "l":
[0.098972, 0.048634, -0.081232]

Final "l":
[-0.301591, -1.567778, 1.092344]
```

This demonstrates that the vectors are trainable parameters rather than
fixed token representations.

The `"o"` embedding did not change in this experiment because `"o"`,
although present as a target, was never used as an input token.

## Major Learning

Embeddings allow the model to learn continuous representations of
discrete tokens.

The representations are learned indirectly through the prediction
objective.

## Important Exception

Embedding dimensions should not be interpreted as individually
human-readable concepts. Their useful structure emerges from training.

## Short Summary

> The model can now learn the representation of input tokens itself
> rather than treating token IDs as ordinary numerical features.

---

# Future Experiments

## Experiment 15 --- Improved Character Language Model

**Status:** Next

Use the embedding-based context model as the foundation for a more
complete character language model.

The immediate goal is to move beyond the tiny `"hello"` demonstration
while keeping the architecture understandable.

Potential progression:

```text
Text
 ↓
Vocabulary
 ↓
Sequence Dataset
 ↓
Token Embeddings
 ↓
Context Representation
 ↓
Network
 ↓
Logits
 ↓
Softmax + Cross Entropy
```

### Major Learning

Observe how the learned embedding and context representation behave when
the dataset contains more character sequences.

### Short Summary

> Move from the minimal embedding demonstration toward a more complete
> character-level language-model training setup.

---

## Experiment 16 --- Self-Attention

**Status:** Planned

Implement a small single-head self-attention mechanism:

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

Attention allows information from different positions in a sequence to
interact dynamically.

### Important Exception

Attention weights are calculated from the current representations; they
are not fixed lookup values.

### Short Summary

> Attention allows the model to determine which parts of the context are
> important for each position.

---

## Experiment 17 --- Multi-Head Attention

**Status:** Planned

Extend self-attention to multiple heads:

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

Multiple attention heads allow the model to learn different interaction
patterns in parallel.

### Important Exception

More heads do not automatically mean a better model. Head count must be
compatible with the model dimension and architecture.

### Short Summary

> Multi-head attention lets several attention patterns be learned in
> parallel.

---

## Experiment 18 --- Transformer Block

**Status:** Planned

Build a reusable Transformer block:

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

A Transformer block combines attention, residual connections,
normalization, and feed-forward computation.

### Short Summary

> A Transformer block combines the core mechanisms required for a
> Transformer language model.

---

## Experiment 19 --- Tiny Transformer

**Status:** Planned

Stack Transformer blocks:

```text
Text
 ↓
Vocabulary
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

Connect the individual components into a small Transformer architecture.

### Important Exception

This will remain an educational model and will be far smaller than a
production LLM.

### Short Summary

> The Transformer combines our previously implemented components into a
> complete language-model architecture.

---

## Experiment 20 --- Tiny Language Model

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

Observe how a complete language model behaves during training and
generation.

### Important Exception

Low training loss does not automatically imply useful language
generation or generalization.

### Short Summary

> A complete tiny language model connects the individual mechanisms to
> actual text generation.

---

## Experiment 21 --- Sampling

**Status:** Planned

Compare:

```text
Greedy decoding
Temperature
Top-K
Top-P
```

### Major Learning

Generation converts probability distributions into token choices.

### Important Exception

Sampling parameters affect diversity and consistency; there is no
universally best setting.

### Short Summary

> Sampling controls how model probabilities become generated text.

---

## Experiment 22 --- Overfitting

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

Low training loss alone is not sufficient evidence of generalization.

### Short Summary

> Training performance and generalization are different things.

---

## Experiment 23 --- NumPy Comparison

**Status:** Planned

Compare selected pure-Python implementations with NumPy.

Measure:

```text
Execution time
Memory usage
Implementation complexity
```

### Major Learning

Understand why optimized numerical libraries are useful for vectorized
computation.

### Important Exception

Optimization should follow measurement rather than assumptions.

### Short Summary

> NumPy shows how optimized numerical operations affect implementation
> and performance.

---

## Experiment 24 --- PyTorch Comparison

**Status:** Planned

Implement the same basic model using PyTorch.

### Major Learning

Understand what tensors, automatic differentiation, and optimized
training frameworks provide after implementing the underlying concepts
ourselves.

### Important Exception

The framework should complement understanding of the mathematics rather
than replace it.

### Short Summary

> PyTorch should make more sense after the underlying mechanisms have
> already been built from scratch.

---

## Experiment 25 --- C++ Implementation

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

C++ is not automatically faster for every workload; implementation and
workload matter.

### Short Summary

> Reimplement a selected computational core to understand systems-level
> performance.

---

## Experiment 26 --- Rust Implementation

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

## Experiment 27 --- GPU Exploration

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
01 — Linear Learning                  ✅
02 — Linear Model Limitation          ✅
03 — Bias                             ✅
04 — Multiple Neurons                 ✅
05 — Activation / ReLU                ✅
06 — Multi-Input Neuron               ✅
07 — Layer Gradient Aggregation       ✅
08 — Network / Multiple Layers        ✅
09 — Network-Based Training           ✅
10 — Multi-Layer Training             ✅
11 — Numerical Gradient Check         ✅
12 — Character Language Model         ✅
13 — Context Prediction               ✅
14 — Embedding                        ✅
```

Next:

```text
15 — Improved Character Language Model
```

Planned:

```text
16 — Self-Attention
17 — Multi-Head Attention
18 — Transformer Block
19 — Tiny Transformer
20 — Tiny Language Model
21 — Sampling
22 — Overfitting
23 — NumPy Comparison
24 — PyTorch Comparison
```

Future systems track:

```text
25 — C++ Implementation
26 — Rust Implementation
27 — GPU Exploration
```

---

# Current Architecture

The project has now evolved into two related paths.

## Numeric / General Network Path

```text
Input Vector
      ↓
Network
      ↓
Layer(s)
      ↓
Neuron(s)
      ↓
Activation
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

## Language Model Path

```text
Text
 ↓
Vocabulary
 ↓
Token IDs
 ↓
Sequence Dataset
 ↓
Embedding
 ↓
Flattened Context Vectors
 ↓
Network
 ↓
Logits
 ↓
SoftmaxCrossEntropy
 ↓
Backpropagation
 ↓
Update Network Parameters
 +
Update Embedding Parameters
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

SoftmaxCrossEntropy
  → converts logits into classification loss and gradients

Embedding
  → maps token IDs to learnable vectors

EmbeddedSequence
  → adapts embedding vectors to the existing network input format

Optimizer
  → updates trainable parameters

Trainer
  → coordinates the learning cycle
```

---

# Major Milestone

The project has moved from a basic neural-network implementation to a
working **character-level language-model foundation**.

We have now demonstrated:

```text
Tokenization
     ↓
Context
     ↓
Embedding
     ↓
Prediction
     ↓
Loss
     ↓
Backpropagation
     ↓
Embedding + Network Updates
```

The embedded experiment demonstrated that the representation itself can
change during training:

```text
Initial embedding
      ↓
Training
      ↓
Updated embedding
```

For the `"hello"` experiment:

```text
Initial "l":
[0.098972, 0.048634, -0.081232]

Final "l":
[-0.301591, -1.567778, 1.092344]
```

while `"o"` remained unchanged because it was only used as a target and
was never part of an input context.

This is an important distinction:

> The embedding matrix is part of the model's trainable parameters.

---

# Important Lessons So Far

1. **More epochs do not fix an architecture that cannot represent the
   target relationship.**

2. **More neurons do not automatically make a model deep.** Depth comes
   from stacking layers.

3. **Linear layers without non-linear activation remain effectively
   linear.**

4. **ReLU can block gradients when its raw input is not positive.**

5. **Input and weight dimensions must match.**

6. **Backward propagation must process layers in reverse order.**

7. **The Trainer should coordinate components rather than duplicate their
   mathematical logic.**

8. **Identical inputs with conflicting targets expose a lack of context,
   not necessarily a training failure.**

9. **Context gives the model additional information needed to distinguish
   different occurrences of the same token.**

10. **Token IDs are categorical identifiers, not meaningful continuous
    representations.**

11. **Embeddings provide learnable continuous representations of tokens.**

12. **Only embedding rows used by an input context receive gradients for
    that training example.**

13. **Repeated token IDs must accumulate their gradient contributions into
    the same embedding row.**

14. **Low training loss alone does not guarantee useful language
    generation or generalization.**

---

# Short Summary

The project has progressed through:

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
Training
  ↓
Gradient Checking
  ↓
Character Prediction
  ↓
Context
  ↓
Embeddings
```

We now have the core pieces required to move toward attention:

```text
Character Language Model
        ↓
Learned Embeddings
        ↓
Self-Attention
        ↓
Multi-Head Attention
        ↓
Transformer Block
        ↓
Tiny Transformer
        ↓
Tiny LLM
```

> **The objective is not simply to build a tiny LLM. The objective is to
> understand every major mechanism that makes the LLM learn.**
