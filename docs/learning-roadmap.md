# LLM From Scratch — Development Plan

## 1. Project Goal

Build a small language model from first principles to understand how modern neural networks and LLMs work internally.

The project will intentionally begin without PyTorch, TensorFlow, Hugging Face, or other ML frameworks.

Learning path:

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
Character Prediction
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

# 2. Repository Strategy

## Public Repository

Recommended: **YES**

Keep this learning project public.

Purpose:

- Document the learning journey
- Demonstrate understanding of ML/LLM fundamentals
- Build a useful GitHub portfolio project
- Allow others to reproduce the experiments
- Keep the implementation educational and transparent

Suggested repository name:

```text
llm-from-scratch
```

The repository should contain only generic learning code and public datasets/examples.

## Private Repository

Keep the production NDC/RxNorm intelligence system private.

Suggested repository:

```text
ndc-rxnorm-intelligence
```

This private project can later contain:

- RxNorm integration
- NDC normalization
- Proprietary matching logic
- Pharmacy/prescription data
- Training datasets
- Evaluation datasets
- Production models
- Business rules
- Customer-specific logic

Important: the public project should teach the underlying technology, while the private project contains the actual healthcare/business intelligence.

---

# 3. Development Principles

1. Start with plain Python.
2. Avoid ML frameworks initially.
3. Implement the mathematics ourselves.
4. Keep every stage small and testable.
5. Add automated tests from the beginning.
6. Record experiments and results.
7. Introduce NumPy only after the pure-Python implementation is understood.
8. Introduce PyTorch only after we understand what it is replacing.
9. Later investigate C++ and Rust for performance/system-level implementations.
10. Never put private healthcare/customer data into the public repository.

---

# 4. Initial Dependencies

## Phase 1 — No Third-Party ML Packages

Required:

```text
Python 3.11+
```

Use only Python standard library initially.

Recommended standard-library modules:

```text
math
random
dataclasses
typing
json
pathlib
```

Development/testing:

```text
pytest
ruff
```

These are development tools, not ML frameworks.

---

# 5. Initial Project Structure

```text
llm-from-scratch/
│
├── README.md
├── LICENSE
├── .gitignore
├── pyproject.toml
│
├── docs/
│   ├── learning-roadmap.md
│   ├── concepts.md
│   └── experiments.md
│
├── src/
│   └── llm_from_scratch/
│       ├── __init__.py
│       │
│       ├── core/
│       │   ├── __init__.py
│       │   ├── neuron.py
│       │   ├── layer.py
│       │   ├── network.py
│       │   ├── loss.py
│       │   ├── gradient.py
│       │   ├── optimizer.py
│       │   └── training.py
│       │
│       ├── data/
│       │   ├── __init__.py
│       │   └── dataset.py
│       │
│       └── language/
│           ├── __init__.py
│           ├── tokenizer.py
│           ├── vocabulary.py
│           ├── embedding.py
│           ├── attention.py
│           └── transformer.py
│
├── experiments/
│   ├── 01_linear_learning.py
│   ├── 02_single_neuron.py
│   ├── 03_multi_neuron.py
│   ├── 04_training_loop.py
│   ├── 05_character_prediction.py
│   ├── 06_tokenization.py
│   ├── 07_embeddings.py
│   ├── 08_attention.py
│   └── 09_tiny_transformer.py
│
├── tests/
│   ├── test_neuron.py
│   ├── test_loss.py
│   ├── test_gradient.py
│   ├── test_optimizer.py
│   ├── test_training.py
│   └── test_tokenizer.py
│
└── notebooks/
    └── experiments.ipynb
```

The notebook is optional. The main implementation should remain normal Python modules so the project stays understandable and testable.

---

# 6. Phase 1 — First Learning Model

## Objective

Teach a model:

```text
y = 3x
```

Example:

```text
x    y
1    3
2    6
3    9
4    12
5    15
```

The model should start with a random weight and learn approximately:

```text
weight = 3
```

without us directly assigning `3`.

---

# 7. Core Classes

## `neuron.py`

### Class

```text
Neuron
```

Responsibilities:

- Store weight
- Store bias
- Perform forward calculation
- Store values required for backward propagation
- Calculate gradients

Conceptual API:

```python
Neuron.forward(input_value)
Neuron.backward(gradient)
Neuron.update(learning_rate)
```

---

## `loss.py`

### Class

```text
MeanSquaredError
```

Responsibilities:

- Calculate prediction error
- Calculate derivative of the loss

Conceptual API:

```python
loss = MeanSquaredError()
loss.forward(prediction, target)
loss.backward(prediction, target)
```

---

## `gradient.py`

### Purpose

Keep gradient-related mathematics separate from the model implementation.

Initial responsibilities:

- Calculate derivatives
- Validate numerical gradients
- Help understand backpropagation

Later this may evolve into a small automatic-differentiation engine.

---

## `optimizer.py`

### Class

```text
GradientDescent
```

Responsibilities:

- Receive parameters and gradients
- Update parameters
- Apply learning rate

Conceptual API:

```python
optimizer.step(parameters, gradients)
```

---

## `layer.py`

### Class

```text
DenseLayer
```

Responsibilities:

- Manage multiple neurons
- Perform forward pass
- Perform backward pass
- Expose parameters and gradients

Conceptual flow:

```text
inputs
  ↓
DenseLayer
  ↓
neurons
  ↓
outputs
```

---

## `network.py`

### Class

```text
NeuralNetwork
```

Responsibilities:

- Manage multiple layers
- Execute forward propagation
- Execute backward propagation
- Expose trainable parameters

Conceptual API:

```python
network.forward(inputs)
network.backward(loss_gradient)
network.parameters()
```

---

## `training.py`

### Class

```text
Trainer
```

Responsibilities:

- Training loop
- Forward pass
- Loss calculation
- Backpropagation
- Parameter updates
- Epoch tracking
- Training metrics

Conceptual workflow:

```text
for epoch:

    prediction = model.forward(input)

    loss = loss_function.forward(
        prediction,
        target
    )

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

# 8. Phase 2 — Multi-Neuron Network

Objective:

Move from:

```text
Input → Neuron → Output
```

to:

```text
Input
  ↓
Dense Layer
  ↓
Multiple Neurons
  ↓
Output
```

Learn:

- vectors
- matrices
- multiple weights
- multiple biases
- activation functions

Add:

```text
activation.py
```

Initial class:

```text
ReLU
```

Later:

```text
Sigmoid
Tanh
Softmax
```

---

# 9. Phase 3 — Character-Level Language Model

Objective:

Teach the model to predict the next character.

Example:

```text
Input:  hel
Target: l
```

Training examples:

```text
h     → e
he    → l
hel   → l
hell  → o
hello → space
```

Add:

```text
language/
├── tokenizer.py
└── vocabulary.py
```

Classes:

```text
CharacterTokenizer
Vocabulary
```

---

# 10. Phase 4 — Tokenization

Move from individual characters toward tokens.

Example:

```text
"amoxicillin 500 mg"
```

could eventually become conceptual tokens such as:

```text
["amoxicillin", "500", "mg"]
```

Learn:

- vocabulary
- token IDs
- special tokens
- encode
- decode

Class:

```text
Tokenizer
```

---

# 11. Phase 5 — Embeddings

Objective:

Convert token IDs into vectors.

```text
Token ID
   ↓
Embedding lookup
   ↓
Vector
```

Class:

```text
Embedding
```

Learn:

- embedding dimensions
- embedding matrix
- learned representations

---

# 12. Phase 6 — Attention

Objective:

Understand the central mechanism behind Transformers.

Conceptual workflow:

```text
Tokens
  ↓
Embeddings
  ↓
Query
Key
Value
  ↓
Attention Scores
  ↓
Weighted Values
  ↓
Context Representation
```

Class:

```text
SelfAttention
```

Initially implement single-head attention.

Only after understanding it move to:

```text
MultiHeadAttention
```

---

# 13. Phase 7 — Transformer Block

Build:

```text
TransformerBlock
```

Components:

```text
Self Attention
      ↓
Residual Connection
      ↓
Layer Normalization
      ↓
Feed Forward Network
      ↓
Residual Connection
      ↓
Layer Normalization
```

Then stack multiple blocks.

---

# 14. Phase 8 — Tiny LLM

Final educational architecture:

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
Linear Output Layer
 ↓
Logits
 ↓
Softmax
 ↓
Next Token
```

Goal:

Generate short text based on learned training data.

This will be a deliberately tiny model, designed for understanding rather than useful production-quality generation.

---

# 15. Training Workflow

Every model should follow the same fundamental cycle:

```text
             TRAINING LOOP

                 Input
                   ↓
              Forward Pass
                   ↓
              Prediction
                   ↓
              Calculate Loss
                   ↓
             Backpropagation
                   ↓
                Gradients
                   ↓
            Update Parameters
                   ↓
             Next Iteration
                   ↓
              Lower Loss
```

Important concepts to learn:

```text
parameter
weight
bias
epoch
batch
learning rate
loss
gradient
backpropagation
gradient descent
activation
embedding
logit
probability
attention
```

---

# 16. Testing Strategy

Every important mathematical component should have tests.

Examples:

```text
Neuron
- forward calculation
- gradient calculation
- parameter update

Loss
- expected loss
- expected gradient

Optimizer
- parameter decreases/increases correctly

Tokenizer
- encode
- decode

Attention
- tensor/vector dimensions
- attention weights
- numerical stability
```

For gradients, eventually use numerical gradient checking:

```text
Analytical gradient
        vs
Numerical gradient
```

This is important because it verifies that our backpropagation implementation is actually correct.

---

# 17. Experiment Tracking

Each experiment should record:

```text
Experiment
Dataset
Model architecture
Learning rate
Epochs
Number of parameters
Initial loss
Final loss
Training time
Result
What we learned
```

Keep short notes in:

```text
docs/experiments.md
```

---

# 18. Package Evolution

Do NOT install everything at the beginning.

## Phase 1

```text
Python standard library
pytest
ruff
```

## Phase 2

Introduce:

```text
NumPy
```

Purpose:

- vectors
- matrices
- vectorized operations
- numerical computation

## Phase 3

Introduce:

```text
PyTorch
```

Purpose:

- tensors
- automatic differentiation
- GPU
- optimized training

Only introduce these after implementing the underlying concepts ourselves.

---

# 19. Later Systems Track

After the Python implementation is understood:

```text
Python implementation
        ↓
NumPy implementation
        ↓
PyTorch implementation
        ↓
C++ implementation
        ↓
Rust implementation
        ↓
CUDA/GPU investigation
```

The goal is to understand both:

```text
AI / mathematics
```

and:

```text
systems / performance
```

---

# 20. Future Project — NDC/RxNorm Intelligence

Keep this in a **separate private repository**.

Possible future architecture:

```text
Prescription
     ↓
Drug/NDC normalization
     ↓
RxNorm
     ↓
Candidate NDCs
     ↓
Matching model
     ↓
Confidence score
     ↓
Matched / Review
```

The public `llm-from-scratch` repository should provide the educational foundation.

The private NDC/RxNorm repository will contain the actual healthcare/business implementation.

Important architectural principle:

```text
RxNorm + deterministic rules
              +
        ML / LLM assistance
```

Do not rely on an LLM alone for medication identity or equivalence decisions.

---

# 21. Future Project — Bhagavad Gita Model

This can eventually become a second project built on the same foundation.

Potential progression:

```text
Gita corpus
    ↓
Clean text
    ↓
Tokenizer
    ↓
Language model
    ↓
Fine-tuning / training experiment
    ↓
Relevant verse retrieval
    ↓
Modern-world interpretation
    ↓
Short story / explanation
```

We should clearly distinguish:

```text
Original scripture
      vs
Model interpretation
      vs
Modern application
```

---

# 22. Definition of Success

The first milestone is NOT:

> Build a powerful LLM.

The first milestone is:

> Build a model whose random parameters become useful parameters through training, and understand exactly why that happened.

Then progressively remove the mystery:

```text
"I can use an LLM"
        ↓
"I understand neural networks"
        ↓
"I understand backpropagation"
        ↓
"I understand embeddings"
        ↓
"I understand attention"
        ↓
"I understand Transformers"
        ↓
"I can build a tiny LLM"
        ↓
"I understand how production LLM systems are assembled"
```

---

# 23. First Development Session

Start with only these files:

```text
src/llm_from_scratch/core/
├── neuron.py
├── loss.py
├── optimizer.py
└── training.py

experiments/
└── 01_linear_learning.py

tests/
├── test_neuron.py
└── test_loss.py
```

First target:

```text
Input → Neuron → Prediction → Loss
                      ↓
                 Backpropagation
                      ↓
               Gradient Descent
                      ↓
                 Updated Weight
```

Dataset:

```text
1 → 3
2 → 6
3 → 9
4 → 12
5 → 15
```

Success criteria:

```text
The model starts with random parameters
and learns approximately:

y = 3x
```

Once this works, stop and document what happened before moving to the next phase.

---

# 24. Suggested Git Commit Sequence

```text
chore: initialize llm from scratch project
feat: implement basic neuron and forward pass
feat: implement mean squared error loss
feat: implement gradient descent optimizer
feat: implement backpropagation
feat: add first training loop
test: add neural network core tests
docs: document first learning experiment
feat: add multi neuron network
feat: add character level language model
feat: add tokenizer and vocabulary
feat: add embeddings
feat: implement self attention
feat: implement transformer block
feat: build tiny language model
```

---

## Final Direction

For now:

```text
PUBLIC
llm-from-scratch
       ↓
Learn everything from first principles

PRIVATE
ndc-rxnorm-intelligence
       ↓
Build proprietary healthcare intelligence

FUTURE
gita-llm
       ↓
Apply the learned LLM technology to the Gita project
```

The public project is the **laboratory**.

The private NDC/RxNorm project is the **product**.

The Gita project can become the **creative application** of everything we learn.
