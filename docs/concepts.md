# LLM From Scratch --- Concepts

This document is a living reference for the core concepts used
throughout the project.

The goal is to understand **what each concept means, why it exists, and
how it contributes to an LLM**.

---

# 1. Machine Learning

Machine learning is the process of allowing a system to learn patterns
from examples rather than explicitly programming every rule.

Traditional programming:

```text
Rules + Input → Output
```

Machine learning:

```text
Input + Expected Output
          ↓
       Learning
          ↓
        Model
          ↓
     New Prediction
```

In this project, the model will start with random parameters and learn
them through training.

---

# 2. Model

A model is a mathematical function with parameters that can be adjusted
during training.

Conceptually:

```text
Input
  ↓
Model
  ↓
Prediction
```

For our first experiment:

```text
y = weight × x + bias
```

The model must learn:

```text
weight ≈ 3
```

for the relationship:

```text
y = 3x
```

---

# 3. Parameter

A parameter is a value inside the model that is learned during training.

Common parameters include:

```text
Weights
Biases
Embedding values
Attention parameters
```

For example:

```text
y = weight × x + bias
```

contains two parameters:

```text
weight
bias
```

A large LLM may contain millions or billions of parameters.

---

# 4. Weight

A weight controls how strongly an input influences a calculation.

Example:

```text
output = weight × input
```

If:

```text
input = 2
weight = 3
```

then:

```text
output = 6
```

During training, weights are adjusted to reduce prediction error.

---

# 5. Bias

A bias is an additional learned value that allows a model to shift its
output.

Example:

```text
output = weight × input + bias
```

Without a bias, the model is more restricted.

With a bias, the model can learn relationships such as:

```text
y = 3x + 2
```

---

# 6. Neuron

A neuron is a small mathematical unit that combines inputs, weights, and
a bias.

Conceptually:

```text
inputs
  ↓
weights
  ↓
weighted sum
  ↓
bias
  ↓
activation
  ↓
output
```

A simple neuron:

```text
z = w × x + b
```

The neuron is one of the fundamental building blocks of a neural
network.

---

# 7. Forward Pass

The forward pass is the process of sending input through the model to
produce a prediction.

```text
Input
  ↓
Weights
  ↓
Calculation
  ↓
Activation
  ↓
Prediction
```

Example:

```text
x = 2
weight = 2.5
bias = 0.5

prediction = 2.5 × 2 + 0.5
           = 5.5
```

The forward pass answers:

> "What does the model currently predict?"

---

# 8. Prediction

A prediction is the output produced by the model for a given input.

Example:

```text
Expected: 6
Predicted: 5.5
```

The difference between the prediction and expected result is used to
calculate the loss.

---

# 9. Loss

Loss measures how wrong the model's prediction is.

Conceptually:

```text
Prediction
     ↓
Compare with target
     ↓
Loss
```

For example:

```text
Expected = 6
Predicted = 5.5
```

The loss function converts this error into a numerical value.

Lower loss generally means the model is performing better on the
training examples.

---

# 10. Mean Squared Error

Mean Squared Error (MSE) is a simple loss function useful for our first
experiments.

For one example:

```text
Loss = (prediction - target)²
```

For multiple examples:

```text
MSE = average((prediction - target)²)
```

We will use MSE initially because it makes the mathematics easy to
understand.

Later, language models will use different loss functions, typically
based on cross-entropy.

---

# 11. Derivative

A derivative tells us how much a value changes when another value
changes.

For machine learning, derivatives help answer:

> "If I change this parameter slightly, what happens to the loss?"

Example:

```text
Loss
  ↑
  │       /
  │      /
  │     /
  │____/________→ Weight
```

The slope tells us the direction in which the loss is changing.

Derivatives are fundamental to backpropagation.

---

# 12. Gradient

A gradient is a collection of derivatives describing how the loss
changes with respect to model parameters.

For example:

```text
Loss
 ↓
Gradient
 ↓
Weight gradient
Bias gradient
```

The gradient tells the optimizer how the parameters should change.

---

# 13. Backpropagation

Backpropagation calculates how much each model parameter contributed to
the final error.

The general process is:

```text
Forward Pass
     ↓
Prediction
     ↓
Loss
     ↓
Backward Pass
     ↓
Gradients
```

The gradients are then used to update the model parameters.

Backpropagation is one of the most important concepts we will implement
ourselves.

---

# 14. Gradient Descent

Gradient descent is an optimization method used to reduce the loss.

Conceptually:

```text
Current parameters
       ↓
Calculate gradient
       ↓
Move parameters
       ↓
Calculate new loss
       ↓
Repeat
```

A simplified update rule:

```text
new_parameter =
    old_parameter - learning_rate × gradient
```

The goal is to move toward a region where the loss is lower.

---

# 15. Learning Rate

The learning rate controls how large each parameter update is.

Too small:

```text
Very slow learning
```

Too large:

```text
Training may become unstable
```

Reasonable learning:

```text
Gradually reduce loss
```

Example:

```text
learning_rate = 0.01
```

The correct value depends on the model and training problem.

---

# 16. Training

Training is the process of repeatedly showing examples to the model and
updating its parameters.

Basic workflow:

```text
Input
 ↓
Forward
 ↓
Prediction
 ↓
Loss
 ↓
Backward
 ↓
Gradient
 ↓
Update parameters
 ↓
Repeat
```

---

# 17. Training Loop

The training loop repeatedly performs the learning process.

Conceptually:

```python
for epoch in epochs:

    prediction = model.forward(input)

    loss = loss_function(prediction, target)

    gradient = loss_function.backward()

    model.backward(gradient)

    optimizer.step()
```

This loop is the core of model training.

---

# 18. Epoch

An epoch represents one complete pass through the training dataset.

Example:

```text
Dataset
100 examples

1 epoch
→ model sees all 100 examples once
```

If we train for:

```text
100 epochs
```

the model processes the dataset 100 times.

---

# 19. Batch

A batch is a group of training examples processed together.

Instead of:

```text
Example 1
Example 2
Example 3
...
```

we can process:

```text
Batch
[Example 1, Example 2, Example 3, ...]
```

Batches become particularly important when training large models on
GPUs.

---

# 20. Activation Function

An activation function introduces non-linearity into a neural network.

Without non-linear activation functions, stacking many linear layers
would still behave essentially like one linear transformation.

Common activation functions:

```text
ReLU
Sigmoid
Tanh
GELU
```

We will initially implement ReLU.

---

# 21. Neural Network

A neural network is a collection of interconnected neurons organized
into layers.

Example:

```text
Input
 ↓
Dense Layer
 ↓
Activation
 ↓
Dense Layer
 ↓
Output
```

A network with multiple layers can learn increasingly complex patterns.

---

# 22. Tensor

A tensor is a generalized container for numerical data.

Examples:

```text
Scalar
1

Vector
[1, 2, 3]

Matrix
[
  [1, 2],
  [3, 4]
]

Higher-dimensional tensor
[
  [
    [...]
  ],
  [
    [...]
  ]
]
```

Modern deep-learning frameworks represent most model data using tensors.

Initially, we will use normal Python data structures to understand the
mathematics.

Later we will introduce NumPy and PyTorch tensors.

---

# 23. Matrix Multiplication

Matrix multiplication is one of the most important operations in neural
networks.

Conceptually:

```text
Input Matrix
      ×
Weight Matrix
      ↓
Output Matrix
```

Large neural networks perform enormous numbers of matrix operations.

This is one major reason GPUs are useful for machine learning.

---

# 24. Token

A token is a unit of text processed by a language model.

Depending on the tokenizer, a token may represent:

```text
Character
Word
Part of a word
Symbol
Number
```

Example:

```text
"learning"
```

might be represented as one token or several smaller tokens depending on
the tokenizer.

---

# 25. Tokenization

Tokenization converts text into tokens.

Conceptually:

```text
"Hello world"
      ↓
["Hello", "world"]
```

The tokens are then converted into numerical IDs:

```text
["Hello", "world"]
       ↓
[1523, 9487]
```

The model operates on numbers rather than raw text.

---

# 26. Vocabulary

A vocabulary is the collection of tokens known to a tokenizer/model.

Example:

```text
Vocabulary

0 → <PAD>
1 → <UNK>
2 → hello
3 → world
4 → drug
5 → pharmacy
```

The vocabulary determines which tokens can be represented directly.

---

# 27. Embedding

An embedding converts a token ID into a vector of numbers.

Conceptually:

```text
Token ID
   ↓
Embedding lookup
   ↓
Vector
```

Example:

```text
"drug"
   ↓
[0.21, -0.73, 0.44, 0.18, ...]
```

The embedding values are learned during training.

Embeddings allow the model to represent relationships between tokens
numerically.

---

# 28. Positional Information

Transformers process tokens in parallel, so they need a way to represent
token position.

For example:

```text
The dog chased the cat
```

has a different meaning/order from:

```text
The cat chased the dog
```

The model therefore needs positional information in addition to token
embeddings.

---

# 29. Attention

Attention allows a model to determine which parts of the input are
important when processing a particular token.

Conceptually:

```text
Token
 ↓
Look at other tokens
 ↓
Calculate relevance
 ↓
Combine useful information
```

For example:

```text
The pharmacy filled the prescription because it had the drug.
```

Attention helps the model determine what words are relevant to
understanding "it".

---

# 30. Query, Key, Value

Self-attention uses three representations:

```text
Query
Key
Value
```

Conceptually:

```text
Query × Key
     ↓
Attention score
     ↓
Weighted Values
     ↓
Context
```

This mechanism allows each token to gather information from other
tokens.

---

# 31. Self-Attention

Self-attention means tokens attend to other tokens within the same
sequence.

Conceptually:

```text
Token 1 ─────┐
Token 2 ─────┼──→ Attention
Token 3 ─────┤
Token 4 ─────┘
```

The result is a context-aware representation of the sequence.

---

# 32. Multi-Head Attention

Instead of performing one attention calculation, a Transformer can
perform several attention operations in parallel.

```text
Input
  ↓
Head 1 ──→ Pattern A
Head 2 ──→ Pattern B
Head 3 ──→ Pattern C
Head 4 ──→ Pattern D
  ↓
Combine
```

Different heads can learn different relationships.

---

# 33. Transformer

A Transformer is a neural-network architecture built around attention
mechanisms.

A simplified Transformer block contains:

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

Transformers are the core architecture behind most modern LLMs.

---

# 34. Logits

Logits are the raw numerical outputs produced by the final model layer
before converting them into probabilities.

Example:

```text
Model output:

hello → 2.1
world → 4.8
drug  → 1.2
```

These values are logits.

They can be converted into probabilities using softmax.

---

# 35. Softmax

Softmax converts logits into a probability distribution.

Example:

```text
Logits

hello → 2.1
world → 4.8
drug  → 1.2

        ↓ Softmax

hello → 0.06
world → 0.88
drug  → 0.06
```

The probabilities add up to approximately:

```text
1.0
```

The model can then select or sample the next token.

---

# 36. Next-Token Prediction

A language model learns to predict the next token based on previous
tokens.

Example:

```text
Input:

"The pharmacy filled the"

Target:

"prescription"
```

During training:

```text
Previous tokens
       ↓
Transformer
       ↓
Probability for next token
       ↓
Compare with actual token
       ↓
Loss
       ↓
Backpropagation
```

This simple objective is the foundation of autoregressive language
models.

---

# 37. Language Model

A language model estimates the probability of tokens given previous
context.

Conceptually:

```text
P(next token | previous tokens)
```

Example:

```text
"The patient took the"
```

Possible predictions:

```text
medication → 0.62
prescription → 0.18
pharmacy → 0.05
...
```

The model learns these probabilities from training data.

---

# 38. Parameter Count

Parameter count is the number of trainable values in a model.

For example:

```text
Model A
10 parameters

Model B
1,000 parameters

Model C
1,000,000 parameters

Model D
1,000,000,000 parameters
```

More parameters generally provide greater capacity, but parameter count
alone does not determine model quality.

Training data, architecture, optimization, and training quality also
matter.

---

# 39. Inference

Inference is using a trained model to produce predictions.

Training:

```text
Data
 ↓
Model
 ↓
Loss
 ↓
Backpropagation
 ↓
Parameter updates
```

Inference:

```text
Input
 ↓
Trained Model
 ↓
Prediction
```

No parameter updates occur during normal inference.

---

# 40. Training vs Fine-Tuning vs RAG

These concepts should remain separate.

### Training

Teach a model from a dataset by updating its parameters.

```text
Dataset
 ↓
Model
 ↓
Parameter updates
```

### Fine-Tuning

Start with an existing pretrained model and continue training it on a
specialized dataset.

```text
Pretrained Model
      ↓
Specialized Dataset
      ↓
Fine-Tuned Model
```

### Retrieval-Augmented Generation (RAG)

Keep knowledge outside the model and retrieve relevant information at
runtime.

```text
Question
   ↓
Retrieve relevant information
   ↓
Model
   ↓
Answer
```

For many real-world applications, RAG is preferable to retraining the
model.

---

# 41. Why We Are Building From Scratch

The project deliberately follows:

```text
Understand
   ↓
Implement
   ↓
Test
   ↓
Compare
   ↓
Use Framework
```

Rather than:

```text
Install Framework
   ↓
Copy Example
   ↓
Run Model
   ↓
Hope It Works
```

The purpose is not to replace mature frameworks.

The purpose is to understand what those frameworks are doing for us.

---

# 42. Our Learning Rule

Whenever we introduce a new abstraction, ask:

> **What problem does this abstraction solve?**

For example:

```text
Neuron
→ Basic computation

Layer
→ Group neurons

Network
→ Combine layers

Loss
→ Measure error

Gradient
→ Determine direction of improvement

Optimizer
→ Update parameters

Embedding
→ Represent tokens numerically

Attention
→ Capture relationships between tokens

Transformer
→ Build scalable contextual representations
```

This principle should guide the entire project.

---

# 43. The Big Picture

Eventually everything connects:

```text
                    TEXT
                     ↓
                 Tokenizer
                     ↓
                  Tokens
                     ↓
                 Token IDs
                     ↓
                 Embeddings
                     ↓
            Positional Information
                     ↓
              Transformer
                     ↓
             Self-Attention
                     ↓
             Feed Forward
                     ↓
                  Logits
                     ↓
                 Softmax
                     ↓
              Next Token
                     ↓
                  TEXT
```

And the learning mechanism underneath is:

```text
Forward Pass
     ↓
Loss
     ↓
Backpropagation
     ↓
Gradient
     ↓
Gradient Descent
     ↓
Updated Parameters
     ↓
Repeat
```

That is the core journey of this project.

---

## Status

Concepts will be expanded as the corresponding implementations are
developed.

Current focus:

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
```

---

# 44. Major Learning, Exceptions, and Quick Review

This section is a compact reference for the most important lessons
established so far.

## Major Learning

### Model

A model is a parameterized mathematical function that learns
relationships from examples.

### Parameter

Weights and biases are learned parameters. Later, embeddings and
attention-related values will also be trainable parameters.

### Neuron

A neuron performs a weighted combination of inputs, adds a bias, and
applies an activation function.

```text
weighted sum + bias
        ↓
    activation
        ↓
      output
```

### Layer

A layer groups neurons that operate on the same input vector.

```text
Input Vector
     ↓
 ┌───┼───┐
 N1  N2  N3
 └───┼───┘
     ↓
 Outputs
```

### Network

A network combines layers sequentially.

```text
Input
  ↓
Layer 1
  ↓
Layer 2
  ↓
Output
```

### Loss

Loss measures how far the prediction is from the target.

### Gradient

A gradient tells us how the loss changes with respect to a parameter.

### Backpropagation

Backpropagation calculates gradients by propagating the error backward
through the computation.

```text
Output
  ↓
Layer 2
  ↓
Layer 1
  ↓
Input
```

### Optimizer

The optimizer uses gradients to update parameters.

```text
new parameter =
    old parameter - learning rate × gradient
```

### Trainer

The Trainer coordinates the complete learning cycle without duplicating
the mathematics owned by the individual components.

```text
Forward
  ↓
Loss
  ↓
Backward
  ↓
Update
```

---

## Important Exceptions and Limitations

### 1. More epochs do not fix an insufficient architecture

If the model cannot represent the target relationship, training longer
does not solve the underlying problem.

### 2. More neurons do not automatically mean a deeper network

Multiple neurons in one layer increase width.

Multiple sequential layers increase depth.

```text
Width:
N1  N2  N3

Depth:
Layer 1
   ↓
Layer 2
```

### 3. Linear layers alone remain linear

Stacking linear transformations without a non-linear activation does not
give the network the expressive power normally associated with deep
neural networks.

### 4. ReLU can block gradients

For our ReLU implementation:

```text
raw input > 0 → gradient passes
raw input ≤ 0 → gradient = 0
```

This behavior is intentional and is part of the activation's backward
pass.

### 5. Input and weight dimensions must agree

For a multi-input neuron:

```text
number of inputs = number of weights
```

A mismatch should be rejected rather than silently producing an
incorrect result.

### 6. Backpropagation order matters

If forward propagation is:

```text
Layer 1 → Layer 2
```

backpropagation must be:

```text
Layer 2 → Layer 1
```

### 7. Low training loss is not enough

A model can achieve low training loss while still failing to generalize
to unseen data.

This becomes important when we study validation data and overfitting.

### 8. Parameter count is not model quality

A larger number of parameters gives a model more capacity, but
architecture, data, optimization, and training quality also matter.

### 9. Training and inference are different

Training changes parameters.

Inference normally does not.

```text
Training:
Input → Prediction → Loss → Backward → Update

Inference:
Input → Prediction
```

---

# 45. Short Concept Summary

The concepts learned so far can be reduced to this progression:

```text
Input
  ↓
Neuron
  ↓
Layer
  ↓
Network
  ↓
Prediction
  ↓
Loss
  ↓
Gradient
  ↓
Backpropagation
  ↓
Optimizer
  ↓
Updated Parameters
```

The architecture then grows toward language:

```text
Numeric Learning
      ↓
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
Tiny Language Model
```

## One-Line Definitions

```text
Neuron
→ Performs a parameterized calculation.

Layer
→ Groups neurons operating on the same input vector.

Network
→ Connects layers sequentially.

Activation
→ Introduces non-linearity and controls gradient flow.

Loss
→ Measures prediction error.

Gradient
→ Measures how the loss changes with respect to parameters.

Backpropagation
→ Calculates gradients through the computation graph.

Optimizer
→ Uses gradients to update parameters.

Trainer
→ Coordinates the learning process.

Token
→ A unit of text processed by a language model.

Embedding
→ A learned vector representation of a token.

Attention
→ Allows representations to incorporate information from other positions.

Transformer
→ Combines attention and feed-forward processing into a scalable architecture.
```

---

# 46. Current Implementation Context

The concepts are being learned alongside a first-principles
implementation.

Current verified architecture:

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

The current implementation has already moved beyond the original
single-neuron model and supports:

```text
Multiple inputs
Multiple neurons
Multiple layers
ReLU activation
Forward propagation
Backward propagation
Gradient aggregation
Network-based training
Weight updates
Bias updates
```

The complete automated test suite currently contains:

```text
28 passed
```

## Current Focus

The immediate learning focus is:

```text
Multi-Layer Training
```

After that, the project will transition from numerical learning into
language-model concepts:

```text
Character Prediction
        ↓
Tokenizer
        ↓
Embeddings
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

---

# 47. Learning Principle

For every new concept, ask three questions:

### What is it?

Understand the definition and mathematics.

### Why do we need it?

Understand the limitation or problem that caused the abstraction to be
introduced.

### What does it change?

Understand how it affects the model's behavior and training.

For example:

```text
Activation

What?
→ Non-linear transformation.

Why?
→ Linear layers alone cannot represent complex non-linear relationships.

What changes?
→ The network can learn non-linear patterns and gradients are affected by the activation.
```

This principle should remain the foundation of the project.

---

# 48. Current Milestone

We have moved through:

```text
Linear Model
      ↓
Bias
      ↓
Multiple Inputs
      ↓
Multiple Neurons
      ↓
Activation
      ↓
Layer
      ↓
Multiple Layers
      ↓
Network
      ↓
Backpropagation
      ↓
Network-Based Training
```

The next milestone is to demonstrate that the complete system can
**learn through multiple layers**, not merely execute forward and
backward calculations.

> **The goal is not just to make the code work. The goal is to
> understand why every abstraction exists and what problem it solves.**
