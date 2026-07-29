# Lesson 3 – Transformers: The Technology That Changed AI Forever

> **Module:** 1 – Modern AI Engineering
> **Level:** Intermediate → Expert
> **Estimated time:** 75–90 minutes

**Goal:** Understand how GPT, Claude, Llama, Gemini, Qwen, DeepSeek, and Mistral actually
process language internally.

In Lesson 2 we drew a box labelled **"Transformer Layers"** and moved on. This lesson
opens that box.

---

## 🎯 Why This Lesson Matters

If you interview at Microsoft, Google, Amazon, Thomson Reuters, Andela, OpenAI, or
Anthropic, one of the most common questions is:

> **"Explain the Transformer architecture."**

Many candidates **memorise** the architecture. Very few actually **understand** it.

The difference shows up immediately in follow-up questions:

- *"Why divide by the square root of d_k?"*
- *"Why can't GPT see future tokens?"*
- *"Why does doubling the context length quadruple the compute?"*

Memorisation collapses on those. Understanding doesn't. By the end of this lesson you'll
be able to answer all three.

---

## 1. Before Transformers: RNNs, LSTMs, and GRUs

Before 2017, AI processed text with:

```text
RNN (Recurrent Neural Network)  →  LSTM  →  GRU
```

These read text **one word at a time**, each word waiting for the one before it:

```text
I  ──►  love  ──►  Artificial  ──►  Intelligence
```

### Problem 1 – It's Slow

Imagine reading a book where you cannot start page 50 until page 49 is completely
finished. That's an RNN. Because each step depends on the previous step's output, the
work **cannot be parallelised** — and GPUs are machines built almost entirely for doing
things in parallel.

This was the real bottleneck. It wasn't that RNNs couldn't learn; it was that you could
never afford to train one big enough.

### Problem 2 – Forgetting Long Context

> The boy who lived in London with his grandparents after travelling around Europe for
> five years finally bought ______

By the time the model reaches the final word, the signal from "boy" has passed through
dozens of transformation steps and has largely faded. This is the **long-term dependency
problem** — also known as the vanishing gradient problem.

LSTMs and GRUs added gates to help information survive longer. They improved things. They
didn't solve them.

### In Simple Terms

Older AI models read a sentence word by word, in order. If the sentence got long, the
model forgot how it started. This caused real problems in translation and in anything
involving long documents.

---

## 2. Then Came the Transformer (2017)

Google published one of the most influential papers in the history of computing:

> ### 📄 *Attention Is All You Need* (Vaswani et al., 2017)

Every major LLM in use today — GPT, Claude, Llama, Gemini, Mistral, Qwen, DeepSeek — is
built on this architecture.

### The Biggest Idea

Instead of reading words one at a time, the Transformer reads **the whole sequence at
once**:

```text
        ┌───────────────────────────────────┐
        │  The   cat   sat   on   the   mat │   ← all processed together
        └───────────────────────────────────┘

              instead of

        The ──► cat ──► sat ──► on ──► the ──► mat
```

Every token is processed **in parallel**. This one change is why training on
internet-scale data became possible — you can finally use a GPU the way a GPU wants to
be used.

### In Simple Terms

A Transformer can look at all the words in a sentence at the same time. This makes it
much faster to train, more accurate, and far better at handling long documents.

> 💡 **The trade-off nobody mentions:** processing everything at once means the model has
> no inherent sense of word *order*. We have to add that back manually — see
> **Positional Encoding** in section 7.

---

## 3. The Core Idea: Attention

Ask yourself which words matter in this question:

```text
Where   is   the   capital   of   France
                  ▲▲▲▲▲▲▲          ▲▲▲▲▲▲
```

Your brain focused on **capital** and **France**, and largely ignored "is" and "the". That
focusing mechanism is **attention**.

### A Human Example

> Ahmed gave Ali his laptop because **he** trusted **him**.

Who trusted whom? To resolve "he" and "him", your brain connects words across the
sentence. Attention lets a model learn exactly these relationships — and crucially, it
connects any two words **in a single step**, no matter how far apart they are.

That's the second breakthrough. In an RNN, connecting word 1 to word 50 requires 50
sequential steps. In a Transformer, it's one operation.

---

## 4. Self-Attention

Now imagine every word asking the same question:

> *"Which other words should I pay attention to?"*

```text
The   cat   drank   milk
```

The word **drank** looks at **cat** (who drank?) and **milk** (drank what?). It cares very
little about **The**.

It's called **self**-attention because the words attend to other words in the **same**
sequence.

### In Simple Terms

Every word looks at every other word and decides which ones are most important for
understanding itself. That is self-attention.

---

## 5. Query, Key, and Value (Q, K, V)

This is the single most-asked Transformer interview topic. It sounds harder than it is.

### The Library Analogy

You walk into a library and ask:

> *"I want books about AI."*

| Component | In the library | In the model |
| --- | --- | --- |
| **Query (Q)** | Your request: *"books about AI"* | What this token is **looking for** |
| **Key (K)** | The label on each book's spine | What each token **offers** |
| **Value (V)** | The actual content inside the book | The information **passed forward** |

```text
Query  ──►  compare against all Keys  ──►  best matches  ──►  return their Values
```

### The AI Version

Every token produces three vectors by multiplying its embedding by three **learned**
weight matrices:

```text
Q = X · W_q          W_q, W_k, W_v are learned during training
K = X · W_k          They are NOT hand-designed.
V = X · W_v
```

> ⚠️ **The mistake that fails interviews:** thinking Q, K, and V are manually created, or
> that they're three different things. They are **three different projections of the same
> input**, learned by gradient descent like every other weight.

### The Actual Formula

You should be able to write this from memory:

```text
                    ⎛  Q · Kᵀ  ⎞
Attention(Q,K,V) = softmax⎜ ──────── ⎟ · V
                    ⎝    √d_k   ⎠
```

Step by step:

| Step | Operation | What it does |
| --- | --- | --- |
| 1 | `Q · Kᵀ` | Score every token against every other token |
| 2 | `÷ √d_k` | **Scale** the scores (explained below) |
| 3 | `softmax` | Turn scores into weights that sum to 1 |
| 4 | `· V` | Weighted average of the Values |

### 🔑 Why Divide by √d_k?

This is the follow-up question that separates memorisation from understanding.

The dot product `Q · Kᵀ` sums over `d_k` dimensions. The more dimensions, the larger the
scores grow. With `d_k = 64`, scores can easily reach the hundreds.

Feed large numbers into softmax and it **saturates** — one value gets ~1.0 and everything
else gets ~0.0. The gradient then becomes almost zero, and the model stops learning.

Dividing by `√d_k` keeps the variance of the scores around 1, so softmax stays in a
range where gradients still flow. It is a numerical stability fix, and it is essential.

### Example — Resolving an Ambiguous Word

```text
The dog chased the cat because it was fast.
```

What does **it** refer to? The token `it` produces a Query, compares it against the Keys
of every other token, and attention weights determine whether "dog" or "cat" contributes
more to its representation. The model learned this behaviour from data — nobody
programmed a pronoun rule.

---

## 6. Self-Attention in Actual Code

Here is a complete, working self-attention implementation in pure Python — no libraries.
**This code and its output are real, not illustrative.**

```python
import math

def softmax(scores):
    m = max(scores)                          # subtract max for numerical stability
    exps = [math.exp(s - m) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


def attention(Q, K, V, mask=False):
    d_k = len(K[0])
    outputs, weights = [], []
    for i, q in enumerate(Q):
        # 1. score this query against every key,  2. scale by sqrt(d_k)
        scores = [sum(a * b for a, b in zip(q, k)) / math.sqrt(d_k) for k in K]

        # causal mask: a token may not look at the future
        if mask:
            scores = [s if j <= i else float("-inf") for j, s in enumerate(scores)]

        w = softmax(scores)                  # 3. normalise into weights summing to 1
        weights.append(w)
        # 4. weighted average of the values
        outputs.append([sum(w[j] * V[j][d] for j in range(len(V)))
                        for d in range(len(V[0]))])
    return outputs, weights


tokens = ["The", "cat", "drank", "milk"]
E = {
    "The":   [0.1, 0.0, 0.0, 0.1],
    "cat":   [1.0, 0.2, 0.0, 0.0],
    "drank": [0.8, 0.1, 0.9, 0.0],
    "milk":  [0.0, 0.1, 1.0, 0.2],
}
X = [E[t] for t in tokens]
out, w = attention(X, X, X)
```

**Real output:**

```text
SELF-ATTENTION WEIGHTS (rows = query token, cols = attended token)
             The     cat   drank    milk
The        0.246   0.256   0.253   0.246
cat        0.200   0.320   0.287   0.192
drank      0.168   0.243   0.335   0.254
milk       0.191   0.191   0.298   0.320
```

Read the **`drank`** row: it gives the *lowest* weight to `The` (0.168) and higher weights
to `cat` (0.243) and `milk` (0.254) — exactly the behaviour we predicted in section 4.
Every row sums to **1.0**.

> ⚠️ **Honest caveat:** these embeddings were hand-picked to make the effect visible, and
> real Q/K/V use learned projection matrices rather than the raw embeddings. The
> **mechanism** is genuinely what runs inside GPT; the numbers are a teaching aid.

---

## 7. Causal Masking — Why GPT Can't See the Future

Your draft of this topic probably skipped this, and most tutorials do. It's the mechanism
that makes GPT work at all.

If a model is being trained to **predict the next token**, and it can see the whole
sentence at once, then predicting the next word is trivial — the answer is right there.
The model would learn nothing.

So decoder-only models apply a **causal mask**: position *i* may attend to positions
`0…i`, but never to anything after it. Future scores are set to `-∞`, which softmax turns
into exactly 0.

Run the same code with `mask=True`:

```text
CAUSAL (MASKED) ATTENTION - each token sees only itself and the past
             The     cat   drank    milk
The        1.000   0.000   0.000   0.000
cat        0.385   0.615   0.000   0.000
drank      0.225   0.326   0.449   0.000
milk       0.191   0.191   0.298   0.320
```

Notice the **triangular shape**. `The` can only see itself. `cat` sees `The` and itself.
Only the final token sees everything.

| Model type | Masking | Consequence |
| --- | --- | --- |
| **BERT** (encoder) | No mask — bidirectional | Great at *understanding*; cannot generate text |
| **GPT** (decoder) | Causal mask | Great at *generating*; each token only sees the past |

**This is why BERT can't write and GPT can't do true bidirectional analysis.** It's one
line in the attention function.

---

## 8. Multi-Head Attention

One attention mechanism is good. Several running **in parallel** are better.

Imagine four people reading the same legal document:

```text
Head 1 → dates          Head 2 → people
Head 3 → locations      Head 4 → legal terms
```

Each finds different relationships. Then all their findings are concatenated and combined.

That's **Multi-Head Attention**.

### The Key Detail Interviewers Probe

Multi-head attention is **not** more expensive than single-head. The embedding dimension
is **split** across the heads:

```text
d_model = 512,  8 heads   ──►   each head works with 512 / 8 = 64 dimensions
                          ──►   concatenate the 8 outputs back to 512
```

So you get 8 different "perspectives" for roughly the cost of one. That's why it's such a
good idea.

### In Simple Terms

Each attention head focuses on something different — one on grammar, one on meaning, one
on relationships, one on context. Together they build the full understanding.

---

## 9. Positional Encoding

Transformers process all tokens simultaneously. That creates a problem:

> **How does the model know which word came first?**

Because without it, these are identical:

```text
Dog bites man   ≠   Man bites dog
```

To a pure attention mechanism, both are just *the same bag of three tokens*. Word order
carries meaning, and we've thrown it away.

The fix is **positional encoding** — giving every token an "address" that is added to its
embedding:

```text
Dog → position 1      bites → position 2      man → position 3
```

The original paper used fixed sine and cosine waves of different frequencies. This has a
neat property: the model can learn *relative* offsets, because the encoding for position
*p + k* is a linear function of the encoding for position *p*.

### What Modern Models Actually Use

Most current models — Llama, Qwen, Mistral, DeepSeek — have moved to **RoPE (Rotary
Positional Embedding)**, which *rotates* the Q and K vectors by an angle based on
position. It handles relative positions more naturally and extends to longer contexts far
better.

> 💡 **Interview edge:** if you mention that modern models use RoPE rather than the 2017
> sinusoidal encoding, you immediately signal that you've read past the original paper.

---

## 10. Feed-Forward Network

After attention, each token passes through a small independent neural network:

```text
Linear (d_model → 4 × d_model)  →  activation  →  Linear (4 × d_model → d_model)
```

**Division of labour inside a Transformer block:**

| Component | Role |
| --- | --- |
| **Attention** | Moves information *between* tokens — "what should I look at?" |
| **Feed-forward** | Processes each token *independently* — "what do I make of it?" |

Interestingly, the feed-forward layers hold roughly **two-thirds of the model's
parameters**. Attention gets the attention, but most of the storage capacity — most of
the "knowledge" — lives in these unglamorous layers.

---

## 11. Residual Connections

Deep networks lose information as it flows through many layers, and gradients vanish
before reaching the early layers.

**Residual (skip) connections** solve this by adding the input back to the output:

```text
        ┌──────────────────────────────┐
        │                              │  (the shortcut)
Input ──┴──►  Sublayer  ──►  (+)  ──►  Output

                 Output = Input + Sublayer(Input)
```

That `+` matters enormously. It gives gradients a direct path back to the earliest layers,
which is what makes 100-layer networks trainable at all. Without residuals, deep
Transformers simply do not train.

A useful way to think about it: each layer doesn't *replace* the representation, it makes
a small **edit** to a running total.

---

## 12. Layer Normalization

Imagine comparing two classes of exam results:

```text
Class A scores:   5 – 20
Class B scores:  80 – 100
```

Comparing them directly is meaningless. Normalization rescales values to a consistent
range so training stays stable and activations don't explode or vanish.

### Two Modern Refinements Worth Knowing

**Pre-norm vs. post-norm.** The 2017 paper normalised *after* the sublayer. Modern models
normalise *before* it:

```text
Post-norm (2017):   x + Sublayer(x)   then normalise    ← harder to train when deep
Pre-norm (modern):  x + Sublayer(Norm(x))               ← much more stable
```

Pre-norm is a major reason very deep LLMs train reliably today.

**RMSNorm.** Llama, Qwen, Mistral and others replaced LayerNorm with RMSNorm, which skips
the mean-centering step. Slightly cheaper, works just as well.

---

## 13. The Full Transformer Block

```text
                Input Tokens
                     │
                     ▼
                 Embeddings
                     │
                     ▼
            Positional Encoding
                     │
        ┌────────────┼─────────────┐
        │            ▼             │
        │  Multi-Head Self-Attention│
        │            │             │
        └───────────►(+)           │  ← residual connection
                     │             │
                     ▼             │
              Layer Normalization  │
                     │             │
        ┌────────────┼─────────────┘
        │            ▼
        │   Feed-Forward Network
        │            │
        └───────────►(+)              ← residual connection
                     │
                     ▼
              Layer Normalization
                     │
                     ▼
            Next Transformer Layer
                     │
                     ⋮   (repeated N times)
                     │
                     ▼
              Output: probability
              over the vocabulary
```

A modern LLM repeats this block dozens — sometimes over a hundred — times. That repetition
*is* the model. There is no other secret ingredient.

---

## 14. Encoder vs. Decoder

The original Transformer had two halves:

| Architecture | Attention | Best At | Examples |
| --- | --- | --- | --- |
| **Encoder-only** | Bidirectional (no mask) | *Understanding* — classification, search, embeddings | BERT, RoBERTa, most embedding models |
| **Decoder-only** | Causal (masked) | *Generating* — chat, completion, code | GPT, Claude, Llama, Qwen, Mistral |
| **Encoder–decoder** | Both | *Transforming* — translation, summarisation | T5, BART |

### Why GPT Is Called "Decoder-Only"

GPT's task is to predict the next token. It needs the causal mask (so it can't cheat) and
it doesn't need a separate encoder to digest a source sequence first — the prompt and the
generated text live in one continuous stream.

> 🔗 **Connecting back to Lesson 1:** the embedding models we used in the RAG pipeline are
> **encoder-only**. The model generating the answer is **decoder-only**. A production RAG
> system runs both architectures, for different jobs. That's a great thing to say in an
> interview.

---

## 15. Two Production Consequences You Must Know

Everything above is theory. Here is where it hits your bill and your latency.

### ⚠️ Attention is O(n²)

Every token attends to every other token. Double the sequence length and the attention
computation **quadruples**:

| Context length | Relative attention cost |
| --- | --- |
| 1,000 tokens | 1× |
| 2,000 tokens | 4× |
| 8,000 tokens | 64× |
| 32,000 tokens | 1,024× |

**This is why long context is expensive**, and why "just stuff everything into the prompt"
is not a strategy. It's also the direct architectural reason RAG exists: retrieve the few
relevant chunks instead of paying quadratically to include everything.

*(FlashAttention and similar techniques dramatically improve the memory and speed
constants, but the fundamental quadratic relationship remains.)*

### ⚡ The KV Cache

Recall from Lesson 2 that generation is autoregressive — one token at a time. Naively,
generating token 500 would mean recomputing attention for all 499 previous tokens. That
would be unusably slow.

Instead, models **cache the Key and Value vectors** of every token already processed:

```text
Generating token 1:  compute K,V for the prompt        ──► cache them
Generating token 2:  compute K,V for token 1 only      ──► append to cache
Generating token 3:  compute K,V for token 2 only      ──► append to cache
```

Two things follow directly, and both matter in production:

1. **Prefill vs. decode are different workloads.** Processing your prompt (prefill) is
   parallel and fast. Generating output (decode) is sequential and slow. This is the
   architectural reason output tokens cost more than input tokens.
2. **The KV cache consumes GPU memory that grows with context length.** On a
   self-hosted deployment, this — not model size — is usually what limits how many
   concurrent users you can serve.

> 💡 **GQA (Grouped-Query Attention)**, used in Llama and Mistral, exists specifically to
> shrink the KV cache by letting several query heads share one set of K/V heads. If you
> ever wondered why an architecture paper obsesses over a caching detail, this is why.

---

## 16. The Modern Transformer ≠ the 2017 Paper

Interviewers at Meta, Mistral, and Alibaba notice when you know this. Today's models keep
the core idea but have replaced most of the components:

| Component | 2017 Original | Modern (Llama / Qwen / Mistral) |
| --- | --- | --- |
| Positional encoding | Sinusoidal | **RoPE** (rotary) |
| Normalization | LayerNorm, post-norm | **RMSNorm**, pre-norm |
| Attention | Multi-head (MHA) | **GQA / MQA** (smaller KV cache) |
| Activation | ReLU | **SwiGLU** |
| Architecture | Encoder–decoder | **Decoder-only** |

**What has not changed:** scaled dot-product attention, residual connections, and stacking
the same block many times. The skeleton from 2017 is still the skeleton.

---

## 🎤 Interview Questions

**Q1. Why did Transformers replace RNNs?**
> Three reasons. They process tokens in parallel instead of sequentially, so they can
> actually exploit GPUs and be trained at scale. They connect any two tokens in a single
> operation, avoiding the vanishing-gradient problem over long distances. And they scale
> predictably with more data and parameters.

**Q2. What is self-attention?**
> A mechanism where every token computes a weighted sum over all tokens in the sequence,
> with the weights determined by how relevant each token is to it. "Self" means the
> queries, keys, and values all come from the same sequence.

**Q3. What are Query, Key, and Value?**
> Three learned linear projections of the same input. The Query is what a token is looking
> for, the Key is what each token advertises, and the Value is the information passed
> forward. Attention scores Queries against Keys, then returns a weighted mix of Values.

**Q4. Why divide by √d_k?**
> Without it, dot products grow with the dimension, pushing softmax into saturation where
> gradients vanish and learning stalls. Scaling keeps the score variance near 1.

**Q5. Why is Multi-Head Attention useful?**
> Different heads capture different relationship types — syntax, coreference, entities,
> long-range dependencies — simultaneously. And because `d_model` is split across heads,
> it costs roughly the same as single-head attention.

**Q6. Why is positional encoding necessary?**
> Self-attention is permutation-invariant — it has no inherent notion of order, so "dog
> bites man" and "man bites dog" would be identical. Positional encoding injects order
> back into the representation.

**Q7. Why can't GPT see future tokens?**
> A causal mask sets attention scores for future positions to −∞. Without it, next-token
> prediction would be trivial during training and the model would learn nothing useful.

**Q8. Why does doubling context length quadruple attention cost?**
> Attention compares every token against every other token, so the score matrix is n × n.
> Doubling n gives 4× the entries.

**Q9. What is the KV cache and why does it matter?**
> A cache of the Key and Value vectors for already-processed tokens, so each new token
> doesn't recompute the whole sequence. It's why generation is fast, and its
> memory growth with context is usually what limits concurrency on self-hosted models.

---

## ❌ Common Interview Mistakes

- ❌ Saying *"attention is just memory"* — it's a learned weighted average, not storage
- ❌ Thinking Q, K, V are manually created — they're **learned projections**
- ❌ Forgetting why positional encoding is needed
- ❌ Saying GPT uses an encoder — it's **decoder-only**
- ❌ Not being able to explain the `√d_k` scaling
- ❌ Forgetting the causal mask, the thing that makes generation training work
- ❌ Describing the 2017 paper as if it were current architecture
- ❌ Not connecting O(n²) attention to why long context costs so much

---

## 🌍 Real-World Applications

Transformers power ChatGPT, GitHub Copilot, AI search engines, coding assistants,
translation systems, document summarisation, legal and medical assistants, AI agents, and
every enterprise RAG system you'll build in this bootcamp.

---

## 📝 Assignment

Explain these in your **own words**:

1. Why were RNNs replaced by Transformers?
2. What is self-attention?
3. Explain Query, Key, and Value using the library analogy.
4. Why do we need positional encoding?
5. What is the difference between encoder-only, decoder-only, and encoder–decoder Transformers?

**Bonus (harder — these are the follow-ups that separate candidates):**

6. Why do we divide by `√d_k` in the attention formula?
7. What would happen if we removed the causal mask from GPT during training?
8. Your RAG system's latency tripled when you increased retrieved chunks from 5 to 15.
   Explain why, using what you learned about attention complexity.

**Practical:**

9. Copy the attention code from section 6 and run it. Change the embedding for `drank` and
   observe how the attention weights shift.
10. Set `mask=True` and confirm you get the triangular pattern. Explain what each row means.

Save your answers in [`assignments/lesson-03/`](../../assignments/lesson-03/).

---

## 📖 Resources

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — the original 2017 paper
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — the best visual explanation available
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) — the paper implemented line by line
- [Let's build GPT from scratch (Karpathy)](https://www.youtube.com/watch?v=kCc8FmEb1nY) — build a working Transformer in a video
- [RoFormer: Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — the RoPE paper
- [FlashAttention](https://arxiv.org/abs/2205.14135) — how attention is made fast in practice

---

## 📊 Progress Tracker

**Module 1: Modern AI Engineering**

- ✅ Lesson 1 – What a Production AI Engineer Does
- ✅ Lesson 2 – How LLMs Actually Work
- ✅ Lesson 3 – Transformer Architecture
- ⬜ Lesson 4 – Embeddings Deep Dive

**Lessons remaining in Module 1:** 17
**Total bootcamp lessons remaining (approx.):** 117

---

## ➡️ What's Next – Lesson 4

**Embeddings Deep Dive**

- What embeddings are mathematically
- How embedding models are trained
- Sentence vs. token embeddings
- Cosine similarity, Euclidean distance, and dot product
- Why vector search works
- How semantic search powers RAG
- Popular embedding models (SentenceTransformers, BGE, E5, Jina, OpenAI-style)
- Hands-on Python examples and interview questions

This is foundational for building production RAG systems, and one of the most frequently
tested topics in AI engineering interviews.

---

[⬅️ Lesson 2](lesson-02-how-llms-actually-work.md) · [🏠 Course home](../../README.md)
