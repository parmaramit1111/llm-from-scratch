# LLM From Scratch

> Building a small language model from first principles to understand how modern LLMs actually work.

This project is a hands-on implementation of the fundamental building
blocks behind neural networks, Transformers, and small language models.

The goal is not to build a production-scale LLM.

The goal is to understand **what happens underneath the abstractions** by
implementing the important mathematics and training steps ourselves.

---

## Why This Project?

Modern LLM usage can hide the underlying mechanics behind a few API calls.

This project removes that abstraction step by step:

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
Neural Network
     ↓
Embeddings
     ↓
Self-Attention
     ↓
Transformer Block
     ↓
Small Language Model
```

---

## Learning Path

The project progressed through:

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
Neural Network
  ↓
Character Prediction
  ↓
Context Prediction
  ↓
Token Embeddings
  ↓
Self-Attention
  ↓
Positional Embeddings
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

Each major component was implemented and tested before moving to the
next stage.

---

## Current Architecture

The completed educational model follows this pipeline:

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

The Transformer block contains:

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

## Small Language Model

The current model is a deliberately tiny trainable language model.

It was trained on a simple character-level task:

```text
he → l
el → l
ll → o
```

The final experiment successfully learned these mappings with very high
confidence.

Example result:

```text
Input: he
Target: l
Prediction: l

Input: el
Target: l
Prediction: l

Input: ll
Target: o
Prediction: o
```

The final training loss reached approximately:

```text
0.000203
```

This demonstrates the complete learning loop:

```text
Input
  ↓
Forward Pass
  ↓
Prediction
  ↓
Loss
  ↓
Backpropagation
  ↓
Gradients
  ↓
Parameter Updates
  ↓
Improved Prediction
```

---

## Concepts Implemented

The project covers:

- Neurons
- Weights and biases
- Forward propagation
- Loss functions
- Derivatives
- Gradients
- Backpropagation
- Gradient descent
- Learning rate
- Training loops
- ReLU activation
- Multi-layer neural networks
- Character-level prediction
- Context prediction
- Token embeddings
- Positional embeddings
- Self-attention
- Feed-forward networks
- Layer normalization
- Residual connections
- Transformer blocks
- Logits
- Softmax
- Cross-entropy loss
- Next-token prediction

---

## Testing

The mathematical components are covered by automated unit tests.

Testing includes:

- Forward calculations
- Backward calculations
- Parameter gradients
- Input gradients
- Gradient-path verification
- Numerical gradient checking
- End-to-end model behavior

The project uses numerical gradient checking where appropriate to compare
analytical gradients against independently calculated numerical gradients.

---

## Project Structure

```text
llm-from-scratch/
│
├── docs/
│   ├── learning-roadmap.md
│   ├── concepts.md
│   └── experiments.md
│
├── src/
│   └── llm_from_scratch/
│       ├── core/
│       ├── language/
│       └── models/
│
├── experiments/
│
├── tests/
│
├── README.md
├── CONTRIBUTING.md
├── LICENSE
└── pyproject.toml
```

### Source Code

Reusable implementations live under:

```text
src/llm_from_scratch/
```

### Experiments

Learning experiments live under:

```text
experiments/
```

The current small-language-model experiment is:

```text
experiments/12_small_language_model.py
```

### Tests

Automated tests live under:

```text
tests/
```

---

## Development Philosophy

### Build Before Abstracting

Instead of starting with a high-level framework, the project first
implements the underlying mathematics directly.

For example:

```text
Understand matrix operations
        ↓
Implement the operation
        ↓
Test the operation
        ↓
Use it inside a larger component
```

### Understand Before Using Frameworks

The purpose is not to replace PyTorch, NumPy, or other mature tools.

The purpose is to understand what those tools are doing for us.

### Test the Mathematics

Important backward-pass implementations are verified with unit tests and,
where useful, numerical gradient checks.

---

## Project Status

**Small Language Model milestone: Completed**

The project has progressed from individual neural-network components to a
complete trainable Transformer-based language model implemented from
scratch.

The current repository is now focused on:

```text
Understanding
    ↓
Implementing
    ↓
Testing
    ↓
Documenting
```

---

## Repository Scope

This repository is intentionally educational.

It is **not** intended to compete with GPT, Claude, Gemini, Llama, or
other production-scale language models.

The objective is:

> **Understand the machine before using the machine.**

---

## Learning in Public

The repository documents the implementation journey, including:

- Experiments
- Debugging
- Mathematical explanations
- Unit tests
- Architectural decisions
- Lessons learned

The code and documentation are kept intentionally simple so that each
major concept can be studied independently.

---

## License

MIT License

---

## The Goal

Start with:

```text
a few numbers
```

and reach:

```text
a working small Transformer language model
```

while understanding the major steps in between.

**Learn the fundamentals. Build the pieces. Remove the black box.**
