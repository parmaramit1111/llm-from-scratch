# Contributing & Development Guide

This document defines the development conventions for the **LLM From Scratch** project.

The goal is to keep the repository simple, consistent, educational, and easy to understand as the project grows from a tiny Python model into a small Transformer implementation.

---

# 1. Branching Strategy

We use a simple three-level Git workflow:

```text
main
  │
  └── dev
       │
       ├── feature/01-linear-learning
       ├── feature/02-bias
       ├── feature/03-multi-neuron
       └── ...
```

## `main`

`main` represents the stable public version of the project.

Rules:

* Must contain working code.
* Documentation should be reasonably complete.
* No experimental/broken implementation should be merged directly.
* Avoid direct development on `main`.

---

## `dev`

`dev` is the integration branch.

It contains the latest completed development work before it is promoted to `main`.

Rules:

* Feature branches start from `dev`.
* Completed features are merged into `dev`.
* `dev` should remain buildable and testable.
* Experimental work belongs in feature branches.

---

## Feature Branches

Every meaningful development step gets its own feature branch.

Branch naming pattern:

```text
feature/<number>-<short-description>
```

Example:

```text
feature/01-linear-learning
```

---

# 2. Feature Branch Roadmap

The initial development sequence is:

```text
feature/01-linear-learning
feature/02-bias
feature/03-multi-neuron
feature/04-activation-functions
feature/05-training-loop
feature/06-character-prediction
feature/07-tokenizer
feature/08-embeddings
feature/09-self-attention
feature/10-transformer
```

## Feature 01 — Linear Learning

```text
feature/01-linear-learning
```

Goal:

```text
Input
 ↓
Neuron
 ↓
Prediction
 ↓
Loss
 ↓
Gradient
 ↓
Parameter Update
```

Target:

```text
y = 3x
```

---

## Feature 02 — Bias

```text
feature/02-bias
```

Goal:

```text
y = 3x + 2
```

Introduce:

* Bias
* Bias gradient
* Multiple trainable parameters

---

## Feature 03 — Multi-Neuron

```text
feature/03-multi-neuron
```

Goal:

Move from one neuron to a small neural layer.

Introduce:

* Multiple neurons
* Layers
* Vectors
* Multiple parameters

---

## Feature 04 — Activation Functions

```text
feature/04-activation-functions
```

Goal:

Introduce non-linear activation.

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

## Feature 05 — Training Loop

```text
feature/05-training-loop
```

Goal:

Create a reusable training workflow.

```text
Forward
 ↓
Loss
 ↓
Backward
 ↓
Gradient
 ↓
Update
 ↓
Repeat
```

---

## Feature 06 — Character Prediction

```text
feature/06-character-prediction
```

Goal:

Move from numerical data to text.

```text
characters
 ↓
character IDs
 ↓
sequence
 ↓
next-character prediction
```

---

## Feature 07 — Tokenizer

```text
feature/07-tokenizer
```

Goal:

Build a tokenizer and vocabulary.

```text
Text
 ↓
Tokens
 ↓
Token IDs
```

---

## Feature 08 — Embeddings

```text
feature/08-embeddings
```

Goal:

Convert token IDs into learned vectors.

```text
Token ID
 ↓
Embedding Matrix
 ↓
Vector
```

---

## Feature 09 — Self-Attention

```text
feature/09-self-attention
```

Goal:

Implement self-attention from scratch.

```text
Input
 ↓
Query / Key / Value
 ↓
Attention Scores
 ↓
Weighted Values
 ↓
Context
```

Start with single-head attention.

---

## Feature 10 — Transformer

```text
feature/10-transformer
```

Goal:

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
Feed Forward
 ↓
Residual Connection
 ↓
Layer Normalization
```

---

# 3. Feature Branch Workflow

Every feature follows this workflow.

## Step 1 — Start from `dev`

```bash
git checkout dev
git pull origin dev
```

## Step 2 — Create the feature branch

```bash
git checkout -b feature/01-linear-learning
```

## Step 3 — Develop

Implement only the scope of the feature.

Avoid unrelated changes.

## Step 4 — Test

Run:

```bash
pytest
```

Also run:

```bash
ruff check .
```

If formatting is configured:

```bash
ruff format .
```

## Step 5 — Review

Before committing:

```bash
git status
git diff
```

Check:

* No debug code
* No secrets
* No unnecessary files
* Tests pass
* Documentation is updated if needed

## Step 6 — Commit

Use the project's commit convention.

Example:

```bash
git add .
git commit -m "feat: implement linear learning"
```

## Step 7 — Push

```bash
git push -u origin feature/01-linear-learning
```

## Step 8 — Pull Request

Create:

```text
feature/01-linear-learning
        ↓
       dev
```

After review and successful tests, merge into `dev`.

---

# 4. Pull Request Rules

Each PR should represent one logical learning milestone.

A PR should answer:

```text
What did we build?
Why did we build it?
How does it work?
How was it tested?
What did we learn?
```

Avoid combining unrelated features.

For example, this is good:

```text
feature/01-linear-learning
```

This is not recommended:

```text
feature/01-linear-learning-and-transformer-and-tokenizer
```

---

# 5. Commit Convention

Use a simple Conventional Commit style.

Format:

```text
<type>: <description>
```

Common types:

```text
feat      New functionality
fix       Bug fix
test      Tests
docs      Documentation
refactor  Code restructuring
perf      Performance improvement
chore     Tooling/configuration
```

Examples:

```text
feat: implement basic neuron
feat: add gradient descent optimizer
feat: implement backpropagation
test: add neuron gradient tests
docs: document linear learning experiment
fix: correct weight gradient calculation
refactor: separate training loop from model
perf: optimize matrix multiplication
```

Keep commit messages:

* Short
* Specific
* Written in imperative style
* Focused on one logical change

Prefer:

```text
feat: implement mean squared error
```

Instead of:

```text
updated some files
```

---

# 6. Python Version

Target:

```text
Python 3.11+
```

Use modern Python syntax where it improves readability.

Example:

```python
def calculate_loss(
    prediction: float,
    target: float,
) -> float:
    ...
```

---

# 7. Python Style

Follow standard Python conventions.

Primary principles:

* PEP 8
* Clear naming
* Small functions
* Explicit types
* Minimal magic
* Prefer readability over cleverness

---

# 8. Naming Convention

## Classes

Use `PascalCase`.

```python
class Neuron:
    ...

class DenseLayer:
    ...

class NeuralNetwork:
    ...

class GradientDescent:
    ...
```

## Functions

Use `snake_case`.

```python
def calculate_loss():
    ...

def forward():
    ...

def backward():
    ...
```

## Variables

Use descriptive `snake_case`.

```python
learning_rate
prediction
target
gradient
weight
bias
```

Avoid:

```python
x1
x2
tmp
foo
bar
```

unless the variable has a clear mathematical purpose.

---

# 9. Type Hints

Use type hints for public functions and important internal methods.

Example:

```python
def forward(
    self,
    input_value: float,
) -> float:
    ...
```

For collections:

```python
def predict(
    self,
    inputs: list[float],
) -> list[float]:
    ...
```

Type hints should improve understanding, not create unnecessary complexity.

---

# 10. Dataclasses

Use `dataclass` when a class primarily represents structured data.

Example:

```python
from dataclasses import dataclass


@dataclass
class TrainingResult:
    epochs: int
    initial_loss: float
    final_loss: float
```

Avoid using dataclasses simply because they are available.

---

# 11. Docstrings

Public classes and important methods should have concise docstrings.

Example:

```python
class Neuron:
    """A single trainable neuron."""

    def forward(self, input_value: float) -> float:
        """Calculate the neuron output."""
        ...
```

Don't write documentation that merely repeats the function name.

---

# 12. Code Organization

Keep responsibilities separated.

Recommended:

```text
core/
├── neuron.py
├── layer.py
├── network.py
├── loss.py
├── gradient.py
├── optimizer.py
└── training.py
```

Responsibilities:

```text
Neuron
→ Individual trainable computation

Layer
→ Collection of neurons

Network
→ Collection of layers

Loss
→ Measures prediction error

Gradient
→ Gradient calculations

Optimizer
→ Parameter updates

Trainer
→ Training workflow
```

Avoid putting everything into one large file.

---

# 13. Separation of Concerns

A model should not know how the training loop works.

Avoid:

```python
model.train_for_100_epochs()
```

Prefer:

```python
trainer.fit(
    model,
    dataset,
)
```

This keeps:

```text
Model
```

separate from:

```text
Training process
```

---

# 14. Mathematical Code

Mathematical code should favor clarity.

Prefer:

```python
prediction = weight * input_value + bias
```

over:

```python
y = w * x + b
```

inside production implementation code.

However, mathematical notation is encouraged in comments and documentation when explaining the corresponding formula.

Example:

```python
# y = wx + b
prediction = weight * input_value + bias
```

This gives us both:

```text
Mathematical notation
+
Readable code
```

---

# 15. Avoid Premature Abstraction

Do not build a generalized framework before we understand the underlying problem.

For example, don't immediately create:

```text
BaseTensor
AbstractOptimizer
GenericComputationGraph
UniversalLayerFactory
```

unless the experiment actually requires them.

Start simple.

Refactor when a real need appears.

---

# 16. Educational Transparency

This repository is designed for learning.

Prefer explicit code over highly compressed implementations.

Prefer:

```python
weighted_input = weight * input_value
biased_input = weighted_input + bias
prediction = activation(biased_input)
```

over:

```python
return activation(weight * x + b)
```

when the expanded version makes the learning process clearer.

Production optimization can come later.

---

# 17. Comments

Comments should explain **why**, not merely repeat **what**.

Bad:

```python
# Multiply weight by input
weighted_input = weight * input_value
```

Better:

```python
# The weighted input determines how strongly this feature
# contributes to the neuron's output.
weighted_input = weight * input_value
```

Mathematical explanations are encouraged when they help understanding.

---

# 18. Tests

Every core mathematical component should have tests.

Example:

```text
tests/
├── test_neuron.py
├── test_loss.py
├── test_gradient.py
├── test_optimizer.py
└── test_training.py
```

Tests should verify both:

```text
Expected behavior
```

and:

```text
Mathematical correctness
```

---

# 19. Gradient Testing

Gradient calculations deserve special attention.

Where practical, compare:

```text
Analytical gradient
```

against:

```text
Numerical gradient
```

This helps detect errors in backpropagation.

A small gradient error can cause the entire training process to fail.

---

# 20. Experiment Code

Experiment scripts belong in:

```text
experiments/
```

Example:

```text
experiments/
└── 01_linear_learning.py
```

Experiment code may be more verbose than library code because it should make the learning process visible.

For example:

```python
print(f"Epoch: {epoch}")
print(f"Loss: {loss}")
print(f"Weight: {weight}")
```

This is acceptable in experiment scripts.

---

# 21. Production Code vs Experiment Code

### Core implementation

Focus on:

```text
Correctness
Reusability
Testing
Clarity
```

### Experiment scripts

Focus on:

```text
Visibility
Learning
Measurements
Observations
```

Do not over-engineer experiments.

---

# 22. Dependencies

Do not add packages without a reason.

Initial dependencies:

```text
Python standard library
pytest
ruff
```

Later:

```text
NumPy
```

Later:

```text
PyTorch
```

Each dependency should have a documented purpose.

---

# 23. Reproducibility

Experiments involving randomness should use an explicit seed.

Example:

```python
import random

random.seed(42)
```

When an experiment depends on randomness, record:

```text
Random seed
Dataset
Hyperparameters
Model configuration
```

This allows us to reproduce results.

---

# 24. Configuration

Avoid scattering magic values throughout the code.

Instead of:

```python
weight -= 0.01 * gradient
```

prefer:

```python
learning_rate = 0.01

weight -= learning_rate * gradient
```

Later, configuration can be moved into dedicated configuration objects/files if needed.

---

# 25. Error Handling

Keep error handling simple during the early stages.

Validate inputs when invalid values could make debugging difficult.

Example:

```python
if learning_rate <= 0:
    raise ValueError("learning_rate must be positive")
```

Don't hide errors with broad exception handling.

Avoid:

```python
try:
    ...
except Exception:
    pass
```

---

# 26. Performance

Do not optimize prematurely.

The project has two stages:

```text
Stage 1
Understand correctness
```

then:

```text
Stage 2
Measure performance
```

Only optimize after measurement shows a meaningful bottleneck.

Later we will compare:

```text
Python
 ↓
NumPy
 ↓
PyTorch
 ↓
C++
 ↓
Rust
 ↓
GPU
```

---

# 27. Documentation Updates

When a feature changes an important concept, update the relevant documentation.

For example:

```text
Feature
   ↓
Implementation
   ↓
Tests
   ↓
Experiment
   ↓
Documentation
```

Possible files:

```text
README.md
docs/learning-roadmap.md
docs/concepts.md
docs/experiments.md
```

---

# 28. Definition of Done

A feature is considered complete when:

* [ ] Implementation is complete
* [ ] Tests are added
* [ ] Tests pass
* [ ] Code passes Ruff
* [ ] Experiment works
* [ ] Relevant documentation is updated
* [ ] No debug code remains
* [ ] No secrets/private data are committed
* [ ] Git diff has been reviewed
* [ ] Commit message follows project convention
* [ ] Pull request is ready

---

# 29. Recommended Feature Completion Workflow

```text
Create branch
     ↓
Implement
     ↓
Test
     ↓
Experiment
     ↓
Document
     ↓
Review diff
     ↓
Commit
     ↓
Push
     ↓
Pull Request
     ↓
Merge into dev
     ↓
Delete feature branch
```

---

# 30. Branch Lifecycle Example

For the first feature:

```bash
git checkout dev
git pull origin dev

git checkout -b feature/01-linear-learning
```

After implementation:

```bash
pytest
ruff check .
ruff format .
```

Then:

```bash
git add .
git commit -m "feat: implement linear learning"
git push -u origin feature/01-linear-learning
```

Create PR:

```text
feature/01-linear-learning
             ↓
            dev
```

After merge:

```bash
git checkout dev
git pull origin dev

git branch -d feature/01-linear-learning
```

Then start:

```bash
git checkout -b feature/02-bias
```

---

# 31. Release Flow

When a meaningful collection of features is complete:

```text
dev
 ↓
Pull Request
 ↓
main
```

Example:

```text
feature/01
feature/02
feature/03
       ↓
      dev
       ↓
     main
```

`main` should represent a stable milestone of the learning journey.

---

# 32. Golden Rules

Keep these rules simple:

### 1. One feature branch = one learning milestone

```text
feature/01-linear-learning
```

should teach one major concept.

### 2. Understand before abstracting

Don't hide the mathematics behind frameworks too early.

### 3. Correctness before performance

First make it correct.

Then make it fast.

### 4. Measure before optimizing

Use actual measurements.

### 5. Document what we learn

The Git history and documentation should tell the story of the project.

### 6. Keep the public repository safe

Never commit:

```text
Credentials
API keys
Private datasets
Patient information
Customer information
Production configuration
Proprietary business logic
```

### 7. Keep the code boring

Readable code is more valuable than clever code in a learning project.

---

# 33. Project Development Philosophy

The project follows this progression:

```text
Understand
   ↓
Implement
   ↓
Test
   ↓
Experiment
   ↓
Measure
   ↓
Document
   ↓
Optimize
```

And eventually:

```text
Pure Python
      ↓
NumPy
      ↓
PyTorch
      ↓
C++
      ↓
Rust
      ↓
GPU / CUDA
```

The objective is not merely to build an LLM.

The objective is to understand **why it works, how it learns, and what happens underneath the frameworks we normally use.**
