# LLM From Scratch

> Building a small Language Model from first principles to understand how modern LLMs actually work.

This project is my hands-on journey to understand **Large Language Models from the ground up**.

Instead of starting with PyTorch, Hugging Face, or an existing pretrained model, the goal is to first implement the fundamental concepts ourselves using Python and mathematics.

The objective is not to build the world's most powerful LLM.

The objective is to understand **what is happening underneath the abstractions**.

---

## 🎯 Why This Project?

Modern LLM development often looks deceptively simple:

```python
model = load_model(...)
output = model.generate(...)
```

But what actually happens inside the model?

How does a random collection of parameters learn?

How does backpropagation change those parameters?

How does text become numbers?

How do embeddings represent words?

How does attention determine which information matters?

How does a Transformer generate the next token?

This project explores those questions by building the pieces ourselves.

---

## 🧠 Learning Path

The project will evolve progressively:

```text
Python + Mathematics
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
Tiny Language Model
```

Later, the implementations will be compared with established frameworks and lower-level technologies.

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
CUDA / GPU
```

---

## 🚀 First Goal

The first model will be intentionally tiny.

We will teach a randomly initialized model to learn:

```text
y = 3x
```

For example:

```text
1 → 3
2 → 6
3 → 9
4 → 12
5 → 15
```

The important part is that we **do not directly tell the model that the answer is 3**.

The model starts with random parameters and learns through:

```text
Forward Pass
     ↓
Prediction
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

This will be our first demonstration of machine learning from scratch.

---

## 📚 Concepts

Throughout the project we will learn and implement:

* Neurons
* Weights
* Biases
* Forward propagation
* Loss functions
* Derivatives
* Gradients
* Backpropagation
* Gradient descent
* Learning rate
* Epochs
* Batches
* Activation functions
* Neural networks
* Tokenization
* Vocabulary
* Token IDs
* Embeddings
* Positional information
* Attention
* Self-attention
* Multi-head attention
* Transformer blocks
* Logits
* Probability distributions
* Next-token prediction
* Language-model training

---

## 📁 Project Structure

The project will grow incrementally.

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
│       ├── data/
│       └── language/
│
├── experiments/
│
├── tests/
│
└── README.md
```

We intentionally won't implement everything on day one.

Each component will be introduced when we reach that stage of the learning journey.

---

## 🧪 Development Philosophy

### Start simple

The first implementations will use:

```text
Python
+
Python standard library
+
Mathematics
```

No deep-learning framework initially.

### Understand before abstracting

Before using:

```python
loss.backward()
```

we should understand what backpropagation is doing.

Before using:

```python
torch.matmul(...)
```

we should understand the underlying matrix operation.

Before using a Transformer implementation, we should understand attention.

The goal is to remove the **black box**.

---

## 🔬 Experiments

Experiments will be kept separate from reusable implementation code.

Examples:

```text
01_linear_learning.py
02_single_neuron.py
03_multi_neuron.py
04_training_loop.py
05_character_prediction.py
06_tokenization.py
07_embeddings.py
08_attention.py
09_tiny_transformer.py
```

Each experiment should answer:

1. What are we trying to learn?
2. What mathematics is involved?
3. What does the code do?
4. What happened during training?
5. What did we learn?

---

## 🧪 Testing

Mathematical implementations will be tested as we build them.

Particular attention will be given to:

* Forward calculations
* Loss calculations
* Gradients
* Parameter updates
* Tokenization
* Attention calculations

Eventually we will also compare:

```text
Analytical Gradient
        vs
Numerical Gradient
```

to verify our backpropagation implementation.

---

## 🛠️ Technology Roadmap

### Phase 1

```text
Python
```

Build the fundamentals without ML frameworks.

### Phase 2

```text
NumPy
```

Learn vectorized numerical computation.

### Phase 3

```text
PyTorch
```

Compare our implementation with a production-grade deep-learning framework.

### Phase 4

```text
C++
```

Explore low-level performance and memory management.

### Phase 5

```text
Rust
```

Explore high-performance systems programming with memory safety.

### Phase 6

```text
CUDA / GPU
```

Understand how the underlying hardware accelerates the matrix operations used by neural networks.

---

## 🌱 Future Applications

This repository is primarily an educational project, but the knowledge developed here will eventually be applied to separate projects.

### NDC / RxNorm Intelligence

A private healthcare project exploring:

```text
Prescription
     ↓
Drug Normalization
     ↓
RxNorm
     ↓
NDC Relationships
     ↓
Matching Model
     ↓
Confidence Score
```

The actual healthcare data, proprietary algorithms, and business logic will remain in a separate private repository.

### Bhagavad Gita Language Model

A future project exploring how language models can work with the Bhagavad Gita to:

* identify relevant teachings
* explain concepts
* connect teachings with modern situations
* generate short illustrative stories

The project will clearly distinguish between the original source material and model-generated interpretation.

---

## ⚠️ What This Project Is Not

This is **not** intended to compete with GPT, Claude, Gemini, Llama, or other large production models.

It is an educational implementation.

The goal is:

> **Understand the machine before using the machine.**

---

## 📈 Progress

* [ ] Project setup
* [ ] First neuron
* [ ] Forward pass
* [ ] Loss function
* [ ] Gradient calculation
* [ ] Backpropagation
* [ ] Gradient descent
* [ ] Training loop
* [ ] Multi-neuron network
* [ ] Character-level language model
* [ ] Tokenizer
* [ ] Embeddings
* [ ] Self-attention
* [ ] Multi-head attention
* [ ] Transformer block
* [ ] Tiny Transformer
* [ ] Tiny language model
* [ ] NumPy implementation
* [ ] PyTorch comparison
* [ ] C++ exploration
* [ ] Rust exploration
* [ ] GPU/CUDA exploration

---

## 🤝 Learning in Public

This repository is intentionally public.

The goal is to document the process rather than only publish the final result.

Expect:

* experiments
* mistakes
* debugging
* mathematical explanations
* performance comparisons
* architectural decisions
* lessons learned

If you are also learning how LLMs work internally, feel free to explore the experiments and follow along.

---

## 📜 License

MIT License

---

## ⭐ The Goal

Start with:

```text
a few numbers
```

and eventually reach:

```text
a tiny Transformer
```

while understanding every major step in between.

**Learn the fundamentals. Build the pieces. Remove the black box.**
