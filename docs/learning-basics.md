# Learning Basics --- How Does a Model Learn?

Before learning about LLMs, Transformers, attention, or embeddings,
understand one simple idea:

> **A model learns by making a prediction, measuring how wrong it was,
> and adjusting itself to make a better prediction next time.**

That's the entire foundation.

---

## 1. Think of the Model as a Student

Imagine teaching a student this relationship:

```text
1 → 3
2 → 6
3 → 9
4 → 12
5 → 15
```

The student doesn't know the rule yet.

We want the student to discover:

```text
answer = 3 × input
```

A machine-learning model works in a similar way.

---

# 2. Neuron --- The Student Who Makes a Guess

A neuron takes an input and produces a prediction.

For our first example:

```text
prediction = weight × input
```

Suppose:

```text
input  = 3
weight = 2
```

The neuron calculates:

```text
prediction = 2 × 3
           = 6
```

But the correct answer is:

```text
9
```

So the neuron made a mistake.

```text
Input
  3
  ↓
Neuron
  ↓
Weight = 2
  ↓
Prediction = 6
```

### Remember:

> **Neuron = makes the prediction.**

---

# 3. Loss --- How Wrong Was the Prediction?

Now we need to measure the mistake.

We use **Mean Squared Error (MSE)**:

```text
loss = (prediction - target)²
```

Our example:

```text
prediction = 6
target     = 9
```

Therefore:

```text
loss = (6 - 9)²
     = (-3)²
     = 9
```

The loss is `9`.

Try a better prediction:

```text
prediction = 8
target     = 9
```

Then:

```text
loss = (8 - 9)²
     = 1
```

Better.

Perfect prediction:

```text
prediction = 9
target     = 9

loss = 0
```

So:

```text
High loss → Bad prediction

Low loss  → Better prediction

Zero loss → Perfect prediction
```

### Remember:

> **Loss = measures how wrong the model was.**

---

# 4. Gradient --- Which Way Should We Move?

Now we know the model is wrong.

But we have another problem:

> **How should we change the weight?**

This is where the **gradient** comes in.

Imagine standing on a mountain in heavy fog.

You can't see the valley.

But you can feel the slope beneath your feet.

The slope tells you which direction is downhill.

The gradient gives the model similar information:

```text
"Change the parameter in this direction
to reduce the loss."
```

Conceptually:

```text
High Loss
    ●
     \
      \
       ●
        \
         ●
          \
           ●
        Low Loss
```

### Remember:

> **Gradient = tells us which direction to change.**

---

# 5. Optimizer --- Actually Make the Change

Now we know which direction to move.

The **optimizer** changes the model's parameters.

A common rule is:

```text
new parameter =
    old parameter - learning rate × gradient
```

Don't worry about the formula yet.

Understand these three words:

```text
Gradient
→ Which direction?

Learning Rate
→ How big a step?

Optimizer
→ Make the change.
```

### Remember:

> **Optimizer = changes the model's parameters.**

---

# 6. The Complete Learning Process

Put everything together:

```text
             INPUT
               ↓
             NEURON
               ↓
           PREDICTION
               ↓
              LOSS
               ↓
        "How wrong am I?"
               ↓
            GRADIENT
               ↓
       "Which way should I move?"
               ↓
           OPTIMIZER
               ↓
        "Change the weight"
               ↓
         UPDATED MODEL
               ↓
             REPEAT
               ↺
```

Every iteration should ideally reduce the loss.

---

# 7. A Complete Example

Let's start with:

```text
input  = 3
target = 9
weight = 1
```

### Step 1 --- Prediction

```text
prediction = weight × input
           = 1 × 3
           = 3
```

### Step 2 --- Loss

```text
loss = (3 - 9)²
     = 36
```

That's a bad prediction.

### Step 3 --- Gradient

The gradient tells us:

```text
"The weight needs to increase."
```

### Step 4 --- Optimizer

The optimizer increases the weight.

Maybe it becomes:

```text
weight = 1.5
```

Now:

```text
prediction = 1.5 × 3
           = 4.5
```

Loss:

```text
(4.5 - 9)²
= 20.25
```

Loss went from:

```text
36 → 20.25
```

We're improving.

The process continues:

```text
weight       prediction       loss

1.0             3             36
1.5             4.5           20.25
2.0             6              9
2.5             7.5            2.25
3.0             9              0
```

Eventually:

```text
weight ≈ 3
```

The model discovered the relationship:

```text
y = 3x
```

**Nobody directly told it that the weight should be 3.**

That's the fascinating part.

---

# 8. Why Do We Repeat It?

One prediction isn't enough.

We repeat the process many times:

```text
Prediction
    ↓
Loss
    ↓
Gradient
    ↓
Update
    ↓
Prediction
    ↓
Loss
    ↓
Gradient
    ↓
Update
    ↓
...
```

This repeated process is **training**.

---

# 9. What Is an Epoch?

An **epoch** is one complete pass through the training data.

If we have:

```text
1 → 3
2 → 6
3 → 9
4 → 12
5 → 15
```

then one epoch means the model has processed all five examples once.

```text
Epoch 1
Epoch 2
Epoch 3
...
Epoch 100
```

The model keeps adjusting its parameters.

---

# 10. What Is the Learning Rate?

The learning rate controls how big each adjustment is.

Imagine walking downhill.

### Too small

```text
Tiny steps
↓
Very slow learning
```

### Too large

```text
Huge jumps
↓
May jump over the valley
↓
Training becomes unstable
```

### Appropriate

```text
Controlled steps
↓
Gradually reaches lower loss
```

So:

> **Learning rate = size of the learning step.**

---

# 11. Where Does Backpropagation Fit?

Once the model becomes more complicated, we have many parameters.

For example:

```text
Input
 ↓
Neuron 1 ─┐
Neuron 2 ─┼→ Output
Neuron 3 ─┘
```

Now we need to determine:

> How much did each parameter contribute to the final error?

**Backpropagation** calculates those gradients by working backward
through the model.

```text
Forward:

Input
 ↓
Layer 1
 ↓
Layer 2
 ↓
Prediction
 ↓
Loss


Backward:

Loss
 ↓
Layer 2 gradients
 ↓
Layer 1 gradients
 ↓
Parameter gradients
```

### Remember:

> **Backpropagation = calculating how each parameter should change based
> on the error.**

---

# 12. The Four Words to Remember

If you're completely new to machine learning, remember just these:

Concept Simple meaning

---

**Neuron** Makes a prediction
**Loss** Measures the mistake
**Gradient** Says which direction to change
**Optimizer** Actually changes the parameters

Then:

```text
Neuron
  ↓
Prediction
  ↓
Loss
  ↓
Gradient
  ↓
Optimizer
  ↓
Better Model
```

Repeat this thousands or millions of times.

That's **learning**.

---

# 13. And This Is Still the Foundation of an LLM

Our example has:

```text
1 weight
```

A real neural network may have:

```text
Millions
or
Billions
of parameters
```

But the basic idea remains:

```text
Input
 ↓
Model
 ↓
Prediction
 ↓
Loss
 ↓
Backpropagation
 ↓
Gradients
 ↓
Optimizer
 ↓
Updated Parameters
 ↓
Repeat
```

An LLM simply applies these ideas to a **much more complex model and
enormous amounts of text**.

Later we will add:

```text
Tokens
 ↓
Embeddings
 ↓
Attention
 ↓
Transformers
 ↓
Next-token prediction
```

But underneath all of that is still the same fundamental learning loop.

---

# The One-Sentence Summary

> **A machine-learning model learns by making predictions, measuring its
> mistakes, calculating how its parameters should change, updating them,
> and repeating the process until its predictions improve.**

That's the foundation.

Everything we're going to build in **LLM From Scratch** grows from this
idea.

---

# 14. What We Have Implemented So Far

The concepts above are now implemented in the project as a working,
tested Python system.

The architecture has evolved from a single neuron into a multi-layer
trainable network:

```text
Input Vector
    ↓
Neuron
    ↓
Layer
    ↓
Network
    ↓
Loss
    ↓
Backpropagation
    ↓
Gradient Descent
    ↓
Updated Parameters
```

## Neuron

The neuron now supports multiple inputs.

Conceptually:

```text
weighted sum =
    weight1 × input1
  + weight2 × input2
  + ...
  + bias
```

It calculates:

- weighted output
- weight gradients
- bias gradient
- input gradients

## Activation

We introduced ReLU:

```text
ReLU(x) = x    when x > 0
ReLU(x) = 0    when x ≤ 0
```

ReLU is applied after the neuron's weighted sum and also controls
gradient flow during backpropagation.

## Layer

A layer contains multiple neurons that share the same input vector.

```text
Input Vector
      ↓
 ┌────┼────┐
 ↓    ↓    ↓
 N1   N2   N3
 ↓    ↓    ↓
 O1   O2   O3
```

During backward propagation, the layer sends the correct gradient to
each neuron and aggregates their input gradients.

## Network

A network is a sequence of layers.

```text
Input
  ↓
Layer 1
  ↓
Layer 2
  ↓
Output
```

Forward propagation processes layers in order.

Backward propagation processes them in reverse order:

```text
Output Gradient
      ↓
Layer 2
      ↓
Layer 1
      ↓
Input Gradient
```

This is our first working demonstration of backpropagation through
multiple layers.

## Trainer

The Trainer now operates on the complete Network rather than a single
Layer.

Its training cycle is:

```text
Network.forward()
      ↓
Loss.forward()
      ↓
Loss.backward()
      ↓
Network.backward()
      ↓
Update every weight and bias
```

The optimizer still updates one parameter at a time. The Trainer
coordinates all parameters across all layers.

---

# 15. Current Milestone

We have verified the architecture with automated tests.

Current verified components:

```text
Neuron
    ✅ Multiple inputs
    ✅ Multiple weights
    ✅ Weight gradients
    ✅ Bias gradients
    ✅ Input gradients

Activation
    ✅ ReLU forward
    ✅ ReLU backward

Layer
    ✅ Multiple neurons
    ✅ Forward propagation
    ✅ Backward propagation
    ✅ Input-gradient aggregation

Network
    ✅ Multiple layers
    ✅ Forward propagation
    ✅ Reverse backward propagation

Loss
    ✅ Mean Squared Error
    ✅ Loss gradients

Optimizer
    ✅ Gradient Descent

Trainer
    ✅ Network-based training
    ✅ Multiple weights
    ✅ Multiple layers
    ✅ Weight and bias updates

Gradient Checking
    ✅ Neuron weight gradients
    ✅ Neuron bias gradients
    ✅ Layer input gradients
    ✅ Network input gradients
    ✅ Analytical vs numerical verification
```

The full test suite is passing after the gradient-checking milestone.

Gradient checking uses numerical finite differences as an independent
verification path:

```text
Analytical gradient
        vs
Numerical gradient
```

Using a small value `ε`, the numerical gradient is approximated as:

```text
f(x + ε) - f(x - ε)
-------------------
        2ε
```

This gives us an independent way to verify that our backpropagation
implementation is calculating gradients correctly.

This is an important milestone because the project has moved from
explaining the learning loop to actually implementing and independently
verifying that learning loop from first principles.

---
