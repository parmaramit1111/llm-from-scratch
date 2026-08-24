# Contributing & Development Guide

This document defines the development conventions for the **LLM From Scratch** project.

The project is an educational implementation of neural networks,
attention, Transformer components, and a small language model from
first principles.

---

# 1. Branching Strategy

We use a simple three-level Git workflow:

```text
main
  │
  └── dev
       │
       └── feature/<number>-<short-description>
```

## `main`

`main` represents the stable public version of the project.

Rules:

- Must contain working code.
- Documentation should be reasonably complete.
- No experimental or broken implementation should be merged directly.
- Avoid direct development on `main`.

## `dev`

`dev` is the integration branch.

Rules:

- Feature branches start from `dev`.
- Completed features are merged into `dev`.
- `dev` should remain buildable and testable.

## Feature Branches

Use:

```text
feature/<number>-<short-description>
```

Examples:

```text
feature/09-positional-embedding
feature/10-transformer-block
feature/11-small-language-model
```

---

# 2. Feature Branch Workflow

Every meaningful development step follows this workflow.

## Start from `dev`

```bash
git checkout dev
git pull origin dev
```

## Create a feature branch

```bash
git checkout -b feature/<number>-<short-description>
```

## Develop

Implement only the scope of the feature and avoid unrelated changes.

## Test

Run the relevant test suite:

```bash
pytest -v
```

If Ruff is configured:

```bash
ruff check .
ruff format .
```

## Review

Before committing:

```bash
git status
git diff
```

Check:

- No debug code
- No secrets
- No unnecessary files
- Tests pass
- Documentation is updated when needed

## Commit

Use the project's Conventional Commit style:

```bash
git add .
git commit -m "feat: implement <feature>"
```

## Push

```bash
git push -u origin feature/<number>-<short-description>
```

## Pull Request

Create:

```text
feature/<number>-<short-description>
                ↓
               dev
```

After review and successful tests, merge into `dev`.

---

# 3. Pull Request Rules

Each PR should represent one logical learning milestone.

A PR should make clear:

```text
What did we build?
Why did we build it?
How does it work?
How was it tested?
What did we learn?
```

Avoid combining unrelated features into one PR.

---

# 4. Commit Convention

Use a simple Conventional Commit style:

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
feat: implement self attention
feat: add transformer block
feat: add trainable small language model
test: add layer normalization tests
docs: update learning roadmap
fix: correct weight gradient calculation
refactor: simplify training loop
```

Keep commit messages short, specific, and focused on one logical change.

---

# 5. Python Version and Style

Target:

```text
Python 3.11+
```

Follow standard Python conventions:

- Clear naming
- Small functions
- Type hints
- Readable code
- Minimal magic
- Prefer clarity over cleverness

Use `PascalCase` for classes:

```python
class TransformerBlock:
    ...
```

Use `snake_case` for functions and variables:

```python
def calculate_loss():
    ...
```

Use type hints for public functions and important methods.

---

# 6. Code Organization

Keep responsibilities separated.

The main implementation areas are:

```text
src/llm_from_scratch/
├── core/
├── language/
└── models/
```

Use the appropriate area for each concept:

```text
core
→ Reusable neural-network and Transformer components

language
→ Language-specific sequence and embedding utilities

models
→ Complete trainable models
```

Experiments belong in:

```text
experiments/
```

Tests belong in:

```text
tests/
```

Documentation belongs in:

```text
docs/
```

Avoid putting unrelated responsibilities into one large file.

---

# 7. Educational Transparency

This is a learning project.

Prefer explicit code when it makes the mathematics easier to understand.

For example:

```python
weighted_input = weight * input_value
biased_input = weighted_input + bias
prediction = activation(biased_input)
```

is preferable to hiding the complete operation behind unnecessary
abstraction.

Mathematical notation is encouraged in comments and documentation when it
helps explain the implementation.

---

# 8. Mathematical Code

Mathematical code should favor clarity.

Example:

```python
# y = wx + b
prediction = weight * input_value + bias
```

Important mathematical operations should have tests.

Where practical, compare:

```text
Analytical Gradient
        vs
Numerical Gradient
```

to verify backpropagation.

A small gradient error can prevent the entire training process from
learning correctly.

---

# 9. Separation of Concerns

Keep model computation separate from the training workflow.

Prefer:

```python
model.forward(inputs)
loss = model.loss(target)
model.backward(target)
```

with training logic handled by the experiment or training workflow.

Avoid hiding a complete training process inside a model method unless the
abstraction is genuinely useful.

---

# 10. Avoid Premature Abstraction

Do not build a generalized framework before the underlying problem is
understood.

Start with simple implementations.

Refactor only when a real need appears.

The project values:

```text
Correctness
+
Understanding
+
Readability
```

over unnecessary abstraction.

---

# 11. Tests

Every important mathematical component should have tests.

Tests should verify:

```text
Expected behavior
+
Mathematical correctness
```

Important areas include:

- Forward calculations
- Backward calculations
- Parameter gradients
- Input gradients
- Gradient paths
- Shape validation
- Numerical gradient checks where appropriate

Run:

```bash
pytest -v
```

before opening a PR.

---

# 12. Experiment Code

Experiment scripts belong in:

```text
experiments/
```

Experiments should make the learning process visible.

They may include:

```python
print(f"Epoch: {epoch}")
print(f"Loss: {loss}")
```

This is appropriate for experiments even when such output would not belong
in reusable library code.

Each experiment should make clear:

1. What is being learned.
2. What mathematics is involved.
3. What the implementation does.
4. What happened during training.
5. What was learned from the result.

---

# 13. Dependencies

Do not add dependencies without a clear reason.

Keep the project lightweight.

The core learning implementation should remain understandable without
depending on a large machine-learning framework.

When a new dependency is necessary, document its purpose.

---

# 14. Reproducibility

Experiments involving randomness should use an explicit seed where
reproducibility matters.

Example:

```python
import random

random.seed(42)
```

When appropriate, record:

```text
Random seed
Dataset
Hyperparameters
Model configuration
```

---

# 15. Performance

Do not optimize prematurely.

Use this order:

```text
Correctness
    ↓
Testing
    ↓
Measurement
    ↓
Optimization
```

Optimize only after a real bottleneck has been identified.

---

# 16. Documentation

When a feature introduces an important concept, update the relevant
documentation.

Possible files:

```text
README.md
docs/learning-roadmap.md
docs/concepts.md
docs/experiments.md
```

The documentation should reflect the implementation that actually exists,
not an outdated future plan.

---

# 17. Security and Repository Hygiene

Never commit:

```text
Credentials
API keys
Private datasets
Patient information
Customer information
Production secrets
Proprietary business logic
```

Before committing, review:

```bash
git status
git diff
```

---

# 18. Definition of Done

A feature is complete when:

- [ ] Implementation is complete
- [ ] Tests are added where appropriate
- [ ] Tests pass
- [ ] Code quality checks pass when configured
- [ ] Experiment works when applicable
- [ ] Relevant documentation is updated
- [ ] No debug code remains
- [ ] No secrets or private data are committed
- [ ] Git diff has been reviewed
- [ ] Commit message follows project convention
- [ ] Pull request is ready

---

# 19. Release Flow

When a meaningful collection of features is complete:

```text
feature branch
      ↓
     dev
      ↓
 Pull Request
      ↓
    main
      ↓
  Git tag / release
```

`main` should represent a stable milestone of the learning journey.

---

# 20. Project Development Philosophy

The project follows:

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

The objective is not merely to build a language model.

The objective is to understand **why it works, how it learns, and what
happens underneath the frameworks normally used to build machine-learning
systems.**
