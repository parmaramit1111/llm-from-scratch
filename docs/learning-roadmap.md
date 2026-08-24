# LLM From Scratch --- Learning Roadmap

## 1. Project Goal

Build a small language model from first principles to understand how
modern neural networks, attention mechanisms, Transformers, and LLMs
work internally.

The project intentionally avoids ML frameworks during the core learning
process.

The primary goal is not to build a production-quality LLM.

The goal is:

> Understand what happens inside an LLM by implementing the important
> mathematical components ourselves.

---

## 2. Learning Path

The project evolved through the following progression:

```text
Python + Math
      ↓
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
Neural Network
      ↓
Multi-Layer Network
      ↓
Character Prediction
      ↓
Context Prediction
      ↓
Token Embeddings
      ↓
Self-Attention
      ↓
Feed-Forward Network
      ↓
Layer Normalization
      ↓
Residual Connections
      ↓
Transformer Block
      ↓
Small Language Model
```

---

## 3. Development Principles

1.  Start with plain Python.
2.  Implement the mathematics ourselves.
3.  Keep every component small and testable.
4.  Add automated tests for important components.
5.  Use numerical gradient checking where appropriate.
6.  Record meaningful experiments and results.
7.  Understand a concept before introducing higher-level frameworks.
8.  Keep the implementation educational and readable.
9.  Prefer explicit code over unnecessary abstraction.
10. Keep the project focused on understanding LLM fundamentals.

---

## 4. Completed Milestones

### Phase 1 --- Neural Network Foundation

**Status: Completed**

Implemented:

```text
Neuron
  ↓
Layer
  ↓
Network
  ↓
Trainer
  ↓
Gradient Descent
```

The implementation supports:

- Multiple inputs
- Multiple weights
- Biases
- Forward propagation
- Backward propagation
- Weight gradients
- Bias gradients
- Input gradients
- ReLU activation
- Multiple neurons
- Multiple layers
- Parameter updates
- End-to-end training

### Phase 2 --- Gradient Verification

**Status: Completed**

A numerical gradient checker was implemented to independently verify
analytical gradients.

The project compares:

```text
Analytical Gradient
        vs
Numerical Gradient
```

using the central-difference approximation:

```text
f(x + ε) - f(x - ε)
-------------------
        2ε
```

Gradient verification was performed across the neural-network components
to ensure that the backward implementations are mathematically
consistent.

### Phase 3 --- Character-Level Language Model

**Status: Completed**

The project moved from general neural networks to language modeling.

The model learned to predict the next character from previous
characters.

Example:

```text
Input  → Target

h      → e
he     → l
hel    → l
hell   → o
```

This introduced the fundamental language-modeling concept:

```text
Context
   ↓
Prediction
   ↓
Loss
   ↓
Gradient
   ↓
Parameter Update
```

### Phase 4 --- Context Prediction

**Status: Completed**

The character model was extended to use multiple characters as context.

For the training sequence:

```text
hello
```

with a context size of two:

```text
he → l
el → l
ll → o
```

The model successfully learned these context-dependent mappings.

This was an important transition from simple character prediction toward
the concept of contextual language modeling.

### Phase 5 --- Token Embeddings

**Status: Completed**

Token IDs were converted into learned vector representations.

```text
Token ID
   ↓
Embedding Lookup
   ↓
Vector
```

Implemented:

```text
Embedding
EmbeddedSequence
```

The embedding implementation includes:

- Embedding lookup
- Forward propagation
- Backward propagation
- Repeated-token gradient accumulation
- Numerical gradient verification
- Joint embedding/network updates

The embedding-based context experiment successfully learned:

```text
he → l
el → l
ll → o
```

with very high confidence.

### Phase 6 --- Self-Attention

**Status: Completed**

Self-attention was implemented from first principles.

The implementation covers the fundamental attention concepts:

```text
Input
  ↓
Query
Key
Value
  ↓
Attention Scores
  ↓
Softmax
  ↓
Weighted Values
  ↓
Context Representation
```

The attention implementation includes trainable projections and
backpropagation.

The attention language-model experiment demonstrated that the model
could learn context-dependent predictions.

### Phase 7 --- Positional Embeddings

**Status: Completed**

Positional information was introduced so that the model can distinguish
between token content and token position.

Conceptually:

```text
Token Embedding
       +
Position Embedding
       ↓
Positioned Sequence
```

The implementation includes:

- Positional embedding lookup
- Forward propagation
- Backward propagation
- Token embedding gradient propagation
- Position embedding gradient propagation
- Numerical gradient verification

### Phase 8 --- Feed-Forward Network

**Status: Completed**

The Transformer feed-forward component was implemented:

```text
Input
  ↓
Linear Projection
  ↓
ReLU
  ↓
Linear Projection
  ↓
Output
```

The implementation includes:

- Input weights
- Input biases
- Output weights
- Output biases
- ReLU activation
- Forward propagation
- Backward propagation
- Parameter gradients
- Input gradients

Gradient behavior was validated through unit tests.

### Phase 9 --- Layer Normalization

**Status: Completed**

Layer normalization was implemented from first principles.

Conceptually:

```text
Input
  ↓
Mean
  ↓
Variance
  ↓
Normalize
  ↓
Gamma × normalized + Beta
```

The implementation includes:

- Mean calculation
- Variance calculation
- Normalization
- Learnable gamma
- Learnable beta
- Forward propagation
- Backward propagation
- Parameter gradients
- Input gradients
- Numerical gradient verification

### Phase 10 --- Residual Connections

**Status: Completed**

Residual connections were implemented:

```text
Input ──────────────────┐
  │                     │
  ↓                     │
Sublayer                │
  │                     │
  └────────── + ←───────┘
              ↓
           Output
```

The backward implementation preserves both gradient paths.

This introduced the residual architecture used throughout modern
Transformer networks.

### Phase 11 --- Transformer Block

**Status: Completed**

The major Transformer block components were combined.

The implemented architecture is:

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

The Transformer block supports:

- Self-attention
- Residual connections
- Layer normalization
- Feed-forward network
- Forward propagation
- Backward propagation
- Gradient path tracking
- Parameter gradients

The complete block is covered by automated tests.

### Phase 12 --- Small Language Model

**Status: Completed**

The project reached its first complete end-to-end language model.

Architecture:

```text
Token IDs
    ↓
Token Embeddings
    ↓
Positional Embeddings
    ↓
Transformer Block
    ↓
Output Projection
    ↓
Logits
    ↓
Softmax
    ↓
Next-Token Prediction
```

The model includes:

- Token embeddings
- Positional embeddings
- Self-attention
- Feed-forward network
- Residual connections
- Layer normalization
- Transformer block
- Output projection
- Softmax
- Cross-entropy loss
- Backpropagation
- Parameter updates

The complete model is covered by unit tests.

---

## 5. Small Language Model Experiment

The final educational experiment trains the model on:

```text
he → l
el → l
ll → o
```

The model successfully learned all three context-dependent predictions.

Final training result:

```text
Final Loss ≈ 0.000203
```

Predictions:

```text
Input: he
Target: l
Prediction: l
Confidence: ≈ 99.9864%

Input: el
Target: l
Prediction: l
Confidence: ≈ 99.9888%

Input: ll
Target: o
Prediction: o
Confidence: ≈ 99.9638%
```

This confirms that the complete training pipeline is functioning:

```text
Forward Pass
     ↓
Loss
     ↓
Backpropagation
     ↓
Gradients
     ↓
Parameter Updates
     ↓
Improved Predictions
```

---

## 6. Current Architecture

The completed educational architecture is:

```text
                    Token IDs
                       ↓
                Token Embeddings
                       ↓
             Positional Embeddings
                       ↓
              Transformer Block
                       │
          ┌────────────┴────────────┐
          ↓                         ↓
    Self-Attention            Feed-Forward
          ↓                         ↓
       Residual                  Residual
          ↓                         ↓
    LayerNorm                  LayerNorm
          └────────────┬────────────┘
                       ↓
                Output Projection
                       ↓
                     Logits
                       ↓
                    Softmax
                       ↓
              Next-Token Prediction
```

---

## 7. Testing Strategy

Important mathematical components have dedicated unit tests.

Validated areas include:

```text
Neuron                 ✓
Layer                  ✓
Network                ✓
Loss                   ✓
Optimizer              ✓
Trainer                ✓
Activation             ✓
Gradient Checking      ✓
Embedding              ✓
Self-Attention         ✓
Positional Embedding   ✓
Feed-Forward           ✓
Layer Normalization    ✓
Residual Connection    ✓
Transformer Block      ✓
Small Language Model   ✓
```

The project also uses numerical gradient checking for important
backpropagation implementations.

---

## 8. Experiment Tracking

Experiments are maintained under:

```text
experiments/
```

The experiments progressively demonstrate:

```text
Basic Learning
      ↓
Neural Networks
      ↓
Multi-Layer Training
      ↓
Character Prediction
      ↓
Context Prediction
      ↓
Embeddings
      ↓
Attention
      ↓
Positional Information
      ↓
Transformer Components
      ↓
Small Language Model
```

The latest experiment is:

```text
experiments/12_small_language_model.py
```

---

## 9. Repository Structure

The current project is organized around three primary areas:

```text
llm-from-scratch/
│
├── docs/
├── experiments/
├── src/
│   └── llm_from_scratch/
├── tests/
├── README.md
├── CONTRIBUTING.md
├── LICENSE
└── pyproject.toml
```

The implementation lives under:

```text
src/llm_from_scratch/
```

Experiments live under:

```text
experiments/
```

Automated tests live under:

```text
tests/
```

Learning documentation lives under:

```text
docs/
```

---

## 10. Project Status

The original learning objective has been achieved.

The project has progressed from individual mathematical components to a
complete trainable Transformer-based language model implemented from
scratch.

Current milestone:

```text
LLM From Scratch — Small Language Model
                    ↓
                 COMPLETED
```

The repository is now ready for final documentation cleanup, release
preparation, and publication of the completed implementation.

---

## 11. What We Learned

The project progressively removed the abstraction around modern LLMs.

We started with:

```text
Weight
Bias
Neuron
```

and progressed through:

```text
Gradient
Backpropagation
Layer
Network
Embeddings
Attention
LayerNorm
Residuals
Transformer
```

until reaching:

```text
Small Language Model
```

The most important lesson is that an LLM is not a single mysterious
component.

It is a composition of understandable mathematical operations:

```text
Represent
   ↓
Transform
   ↓
Attend
   ↓
Normalize
   ↓
Transform
   ↓
Predict
   ↓
Measure Error
   ↓
Backpropagate
   ↓
Update
```

---

## 12. Next Direction

The `llm-from-scratch` learning milestone is now complete.

Future work should be treated as separate projects or experiments rather
than continuing to expand the original learning roadmap indefinitely.

Potential future directions include:

```text
NumPy implementation
        ↓
PyTorch implementation
        ↓
Larger Transformer
        ↓
Dataset-based training
        ↓
Text generation
```

These are optional extensions rather than requirements for completing
the current project.

---

## 13. Definition of Success

The project succeeded if we can move from:

> "I can use an LLM."

to:

> "I understand how an LLM works internally."

and ultimately:

> "I can implement the fundamental components of a Transformer language
> model and explain how they work together."

That milestone has now been achieved.
