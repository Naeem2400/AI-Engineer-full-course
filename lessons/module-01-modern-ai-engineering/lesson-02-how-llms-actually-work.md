# Lesson 2 – How Large Language Models (LLMs) Actually Work

> **Module:** 1 – Modern AI Engineering
> **Level:** Intermediate → Expert
> **Estimated time:** 60–75 minutes

**Goal:** Understand what actually happens inside ChatGPT, Claude, Llama, Qwen, Mistral,
Gemini, and other modern LLMs — so you stop treating them as magic black boxes.

In Lesson 1 we established that the LLM is **one component** in a larger system. This
lesson opens that component up and looks inside.

---

## 🗺️ Lesson Roadmap

1. What is an LLM?
2. How does an LLM learn?
3. What are tokens?
4. What is tokenization?
5. What are embeddings?
6. What is a context window?
7. Why do LLMs hallucinate?
8. How do LLMs generate answers?
9. Temperature and sampling
10. Comparing GPT, Claude, Llama, Qwen, Mistral, DeepSeek, and Gemma
11. Interview questions
12. Practical exercise

---

## 1. What Is an LLM?

An **LLM (Large Language Model)** is a neural network trained on enormous amounts of text
to predict the **next token** in a sequence.

Two things it does **not** do by default:

- ❌ It does **not** search the internet.
- ❌ It does **not** "understand" language the way a human does.

Instead, it learns statistical patterns from billions or trillions of tokens. Think of it
as an extraordinarily advanced **pattern prediction machine**.

### Example

Given this input:

```text
The capital of France is ______
```

The model predicts:

```text
Paris
```

Why? Because during training it saw that pattern an enormous number of times.

### In Simple Terms

An LLM is a very large AI model trained on billions or trillions of words. Its entire job
is to answer one question, over and over:

> *"What is the most likely next token?"*

It does not think like a human. It learns patterns.

### ⚠️ Important Interview Point

A common misconception:

> ❌ *"ChatGPT knows everything."*

A much better framing:

> ✅ *"An LLM predicts the most likely next token based on its training data and the
> context it has been given."*

That single sentence explains hallucinations, knowledge cutoffs, and why RAG exists. If
you internalise nothing else from this lesson, internalise that one.

---

## 2. How Does an LLM Learn?

Simplified, training looks like this:

```text
Books
Wikipedia
Research Papers
Websites
Documentation
Code
Conversations
           │
           ▼
   Massive Training Dataset
           │
           ▼
   Neural Network Training
           │
           ▼
      Learns Patterns
           │
           ▼
       Trained LLM
```

### The Training Loop

The model is shown text with the next word hidden:

| | |
| --- | --- |
| **Input** | `The sky is` |
| **Correct answer** | `blue` |
| **Model predicts** | `green` ❌ |

The prediction is wrong. So:

1. A **loss** is calculated — a number measuring how wrong the prediction was.
2. **Backpropagation** works out which weights contributed to the error.
3. Billions of **weights are nudged** slightly in the direction that would have been less wrong.

This repeats **trillions of times**. Nobody labels the data by hand — the text itself
provides the answer, because the next word is always right there. This is why it's called
**self-supervised learning**, and it's the reason LLMs could scale to internet-sized
datasets.

### In Simple Terms

During training the AI is given a sentence and asked to guess the next word. If the guess
is wrong, a loss is calculated and the weights are updated. Repeat this millions, then
billions, then trillions of times, and the model gradually learns the patterns of language.

### The Three Training Stages (Worth Knowing for Interviews)

| Stage | What Happens | Result |
| --- | --- | --- |
| **1. Pre-training** | Next-token prediction over a huge text corpus | A model that completes text but doesn't follow instructions well |
| **2. Instruction tuning (SFT)** | Fine-tuning on examples of instructions and good responses | A model that answers questions instead of continuing them |
| **3. Alignment (RLHF / RLAIF / DPO)** | Training on human or AI preferences about which response is better | A model that is helpful, harmless, and honest |

> 💡 This is why a raw pre-trained "base model" behaves very differently from the chat
> models you're used to. Ask a base model *"What is the capital of France?"* and it might
> reply with *"What is the capital of Germany? What is the capital of Spain?"* — because
> statistically, a list of questions often follows a question.

---

## 3. What Are Tokens?

This is one of the most important practical concepts in the entire bootcamp.

Most beginners assume AI reads **words**. It doesn't. It reads **tokens**.

### Example

```text
I love artificial intelligence.
```

might become:

```text
["I", " love", " artificial", " intelligence", "."]
```

But one word can become several tokens:

```text
unbelievable  ──►  ["un", "believ", "able"]
```

Notice that the leading **space is usually part of the token** (`" love"`, not `"love"`).
That surprises people the first time they see it.

### Rough Rule of Thumb (English)

| Measure | Approximate Value |
| --- | --- |
| 1 token | ~4 characters |
| 1 token | ~0.75 words |
| 100 tokens | ~75 words |
| 1,000 tokens | ~750 words (~1.5 pages) |

⚠️ These ratios are for English. Code, numbers, and non-English languages tokenize far
less efficiently — the same sentence in Urdu, Arabic, or Hindi can cost **2–3× more
tokens** than in English. If you build a multilingual product, this hits your cost and
your context budget directly.

### Why Tokens Matter to You as an Engineer

You pay for:

- **Input tokens** (your prompt + retrieved documents + history)
- **Output tokens** (usually more expensive than input)

And every model has a hard limit called the **context window**.

> 💰 **Real consequence:** in a RAG system, retrieved documents are usually the largest
> part of your prompt. Retrieving 20 chunks instead of 5 can quadruple your bill for
> *every single query* — often with no improvement in answer quality. This is why the
> reranking step mentioned in Lesson 1 matters so much.

### Interview Question

**Q: What's the difference between words and tokens?**

> A word is a linguistic unit. A token is the unit the model actually processes. One word
> can map to one or many tokens depending on the tokenizer, and common words are usually
> a single token while rare words get split into several pieces.

---

## 4. Tokenization

Before the model reads your prompt:

```text
Tell me about AI.
```

it converts it into tokens, then into numbers.

### The Full Input Pipeline

```text
    User Prompt        "Tell me about AI."
         │
         ▼
     Tokenizer         splits into pieces
         │
         ▼
      Tokens           ["Tell", " me", " about", " AI", "."]
         │
         ▼
   Token IDs           [41551, 757, 922, 15592, 13]
         │
         ▼
    Embeddings         each ID ──► a vector of numbers
         │
         ▼
     Transformer       the actual neural network
```

**The model never sees letters.** It sees numbers. Every single thing an LLM does is
arithmetic on vectors.

### In Simple Terms

The AI does not directly understand English or Urdu text. First the text becomes tokens,
then those tokens become numbers, then those numbers become vectors, and only then does
the neural network process them.

### Why Tokenizers Explain Weird LLM Behaviour

Several famous "AI is dumb" moments are really tokenizer artefacts:

- **"How many r's are in strawberry?"** — the model may see `["str", "aw", "berry"]`, not
  individual letters. Counting characters is genuinely hard when you can't see characters.
- **Arithmetic errors on long numbers** — `1234567` might split into `["123", "45", "67"]`,
  so digit-place alignment is not obvious to the model.
- **Reversing a string is hard** — again, no direct access to characters.

> These aren't reasoning failures. They're representation failures. Knowing the difference
> is exactly the kind of thing that separates a strong candidate from a weak one.

### Code Example — Counting Tokens Before You Send Them

```python
# pip install tiktoken
import tiktoken

encoder = tiktoken.get_encoding("cl100k_base")

text = "I love artificial intelligence."
tokens = encoder.encode(text)

print(tokens)                          # [40, 3021, 21075, 11478, 13]
print(len(tokens))                     # 5
print([encoder.decode([t]) for t in tokens])
# ['I', ' love', ' artificial', ' intelligence', '.']
```

**Production habit:** count tokens *before* sending a request. It lets you reject
oversized inputs early, estimate cost per request, and avoid a context-overflow error in
front of a real user.

---

## 5. What Are Embeddings?

This is one of the most important topics in enterprise AI, and the foundation of
everything we'll build in the RAG module.

Consider these words:

```text
Doctor    Physician    Hospital    Medicine
```

Their meanings are related. Embeddings convert text into a list of numbers so that
**similar meanings land close together** in a mathematical space.

```text
Doctor     ──►  [ 0.42, -0.13,  1.81, ... ]
Physician  ──►  [ 0.40, -0.11,  1.79, ... ]     ← very close to Doctor
Banana     ──►  [-0.91,  1.44, -0.22, ... ]     ← far away
```

*(Illustrative numbers only — real embeddings typically have hundreds to thousands of
dimensions.)*

### Why This Is Powerful

A user searches for:

```text
heart attack treatment
```

The document says:

```text
myocardial infarction therapy
```

**Zero words in common.** Keyword search finds nothing. Embedding-based search finds it
easily, because the two phrases mean nearly the same thing.

This is the foundation of **semantic search** and **RAG**.

### In Simple Terms

An embedding means converting words into numbers — but not random numbers. Numbers that
represent *meaning*. That is why the AI can tell that "Doctor" and "Physician" are nearly
the same thing, even though they share no letters.

### How Similarity Is Measured

The standard tool is **cosine similarity** — it measures the angle between two vectors,
ignoring their length:

```python
def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    return dot / (norm_a * norm_b)
```

| Score | Meaning |
| --- | --- |
| `1.0` | Identical direction — same meaning |
| `~0.8` | Strongly related |
| `~0.5` | Loosely related |
| `0.0` | Unrelated |
| `-1.0` | Opposite direction |

### ⚠️ Token Embeddings vs. Text Embeddings — Don't Confuse Them

This trips up a lot of candidates:

| | **Token embeddings** | **Text / sentence embeddings** |
| --- | --- | --- |
| **What is embedded** | A single token | A whole sentence, paragraph, or chunk |
| **Where it lives** | *Inside* the LLM, first layer | A *separate* model you call yourself |
| **What you use it for** | Nothing directly — it's internal | Semantic search, RAG, clustering |
| **Example** | GPT's internal embedding matrix | `text-embedding-3-small`, BGE, E5 |

When an AI engineer says "embeddings", they almost always mean the **second** kind.

---

## 6. What Is the Context Window?

The context window is the model's **working memory** — everything it can see at once.

It contains:

```text
      System Prompt          "You are a legal assistant..."
            +
   Conversation History      previous turns
            +
   Retrieved Documents       from your vector database (RAG)
            +
     Current Question        what the user just asked
            │
            ▼
           LLM
            │
            ▼
      Output tokens          also count toward the limit
```

If the total exceeds the model's limit, something **must** be dropped or summarised.

### The Most Important Thing to Understand

> **The context window is not memory. It is re-sent every single time.**

An LLM is **stateless**. It remembers nothing between requests.

When ChatGPT appears to "remember" what you said five messages ago, what actually happens
is that the entire conversation is resent with every new message:

```text
Request 1:  [system] + [Q1]                        ──► LLM
Request 2:  [system] + [Q1] + [A1] + [Q2]          ──► LLM
Request 3:  [system] + [Q1] + [A1] + [Q2] + [A2] + [Q3]  ──► LLM
```

**Two consequences that catch engineers out:**

1. **Cost grows quadratically in long chats.** Every turn resends everything before it.
   A 50-turn conversation is dramatically more expensive than 50 separate questions.
2. **You must manage history yourself.** Real systems use sliding windows, summarisation
   of older turns, or retrieval over past messages. There is no "memory" feature that
   comes for free.

### In Simple Terms

Think of the context window as short-term memory that lasts for exactly one request.
Whatever fits in that memory is what the AI can use. Anything beyond it must be removed
or summarised — and next time, you have to send it all over again.

### 🔍 "Lost in the Middle"

A large context window does **not** mean the model uses all of it equally well. Research
consistently shows models attend most reliably to information at the **beginning** and
**end** of the context, and can miss details buried in the middle.

**Practical takeaway:** put your most important retrieved documents and your actual
instruction near the *start* or *end* of the prompt — not buried in the middle of 40
chunks. Retrieving *less but better* usually beats retrieving *more*.

---

## 7. Why Do LLMs Hallucinate?

**Hallucination** = the model generates information that sounds convincing but is
incorrect or unsupported.

### Example

**User:**
> "Who invented the XYZ-999 protocol?"

**Model:**
> "Dr. John Smith invented it in 2018."

No such protocol exists. The model invented a plausible-sounding answer.

### Why Does This Happen?

Go back to the core definition: the model predicts the **most likely next token**.

It was trained on text where questions are followed by confident answers. "I don't know"
is statistically rare in its training data. So when knowledge is missing, the model still
produces the most *plausible-looking* continuation — which means a fluent, confident,
fabricated answer.

> **Hallucination is not a bug. It is the expected behaviour of a next-token predictor
> operating without grounding.** That reframing is worth a lot in an interview.

Common contributing causes:

- Missing or outdated knowledge (training cutoff)
- Ambiguous or under-specified prompts
- No retrieval from trusted sources
- The model filling gaps with plausible text
- Sampling randomness (see temperature, below)

### How Engineers Reduce Hallucinations

| Technique | What It Does |
| --- | --- |
| **RAG** | Supplies real documents so the model doesn't have to guess |
| **Explicit refusal instructions** | *"If the sources don't contain the answer, say you don't know."* |
| **Require citations** | Forces answers to be traceable to a source |
| **Grounding validation** | Programmatically check the answer against the sources (we built this in the Lesson 1 assignment) |
| **Lower temperature** | Less randomness for factual tasks |
| **Structured output** | Constrain the response to a schema |
| **Evaluation sets** | Measure hallucination rate so you know if it's getting better or worse |

> ⚠️ **No technique eliminates hallucination.** They reduce its rate. Any system design
> that assumes 0% hallucination is a system design that will fail in production. Design
> for detection and graceful failure, not for perfection.

---

## 8. How Does an LLM Generate an Answer?

```text
User Prompt
      │
      ▼
  Tokenizer
      │
      ▼
  Token IDs
      │
      ▼
  Embeddings
      │
      ▼
Transformer Layers      ◄─── Lesson 3 opens this box
      │
      ▼
Probability distribution over EVERY token in the vocabulary
      │
      ▼
  Pick one token
      │
      ▼
 Append to the sequence
      │
      ▼
    Repeat ───────────► until a stop token or the length limit
```

This is called **autoregressive generation** — each new token is produced by feeding the
entire sequence, including everything just generated, back through the model.

### Real-World Example

**User:** *"Write a Python function to reverse a string."*

The model does not produce the program at once. It predicts:

```text
"def"  →  " reverse"  →  "_string"  →  "("  →  "text"  →  ")"  →  ":"  →  ...
```

one token at a time, each conditioned on everything before it.

### Two Things This Explains

1. **Why streaming works.** Tokens are produced sequentially, so they can be sent to the
   user as they appear. That's why ChatGPT "types".
2. **Why output is slower than input.** Input tokens are processed in parallel in a single
   pass. Output tokens must be generated **one at a time**, each requiring a full forward
   pass through the network. This is why generating 1,000 tokens takes far longer than
   reading 1,000 tokens — and why output tokens cost more.

---

## 9. Temperature and Sampling

At each step the model produces a **probability distribution** over its entire vocabulary:

```text
The capital of France is ...

  Paris    → 92%
  Lyon     →  3%
  Marseille→  2%
  the      →  1%
  ...
```

**How you pick from that distribution changes the model's behaviour completely.**

| Parameter | What It Does | When to Use |
| --- | --- | --- |
| **temperature = 0** | Always take the highest-probability token (greedy) | Extraction, classification, factual Q&A, structured output |
| **temperature ~0.7** | Balanced randomness | General chat, explanations |
| **temperature 1.0+** | Flatter distribution, more surprising choices | Creative writing, brainstorming |
| **top_p** (nucleus) | Only sample from the smallest set of tokens covering *p* of the probability mass | Common alternative to temperature |
| **top_k** | Only sample from the *k* most likely tokens | Cruder cap on randomness |

### Production Rule of Thumb

> For **RAG and factual systems, use temperature 0 or very close to it.** You want the
> same question to give the same answer, and you want the least creative choice available.
> Creativity is a liability when a lawyer is reading the output.

⚠️ Note that temperature 0 gives *near*-deterministic output, not *guaranteed* identical
output — batching and floating-point non-determinism on GPUs mean you can still see small
variations. Don't build a system that depends on byte-identical responses.

---

## 10. Comparing the Major Model Families

> ⚠️ Model specifications, prices, and context windows change constantly. **Always check
> the provider's current documentation** for numbers. What follows is the durable,
> structural picture that changes slowly.

| Family | Made By | Weights | Typical Positioning |
| --- | --- | --- | --- |
| **GPT** | OpenAI | Closed (API only) | Broad general-purpose ecosystem, very widely adopted |
| **Claude** | Anthropic | Closed (API only) | Strong on long-context work, coding, and careful instruction-following |
| **Gemini** | Google | Closed (API only) | Strong multimodal support, tight Google Cloud integration |
| **Llama** | Meta | **Open weights** | The most common base for self-hosting and fine-tuning |
| **Mistral** | Mistral AI | Mostly **open weights** | Efficient, small models with strong performance per parameter |
| **Qwen** | Alibaba | **Open weights** | Very strong multilingual and Chinese-language capability |
| **DeepSeek** | DeepSeek AI | **Open weights** | Strong reasoning and coding at a low cost point |
| **Gemma** | Google | **Open weights** | Small models designed to run on modest hardware |

### The Decision That Actually Matters: Closed API vs. Open Weights

This is the architectural choice you'll be asked to justify in interviews — not
"which model is best".

| | **Closed API** (GPT, Claude, Gemini) | **Open weights** (Llama, Qwen, Mistral, DeepSeek, Gemma) |
| --- | --- | --- |
| **Setup** | An API key, working in minutes | Provision GPUs, serve, monitor, scale |
| **Cost model** | Per token — scales with usage | Per GPU-hour — fixed whether busy or idle |
| **Data privacy** | Data leaves your infrastructure | Data can stay entirely in your network |
| **Customisation** | Prompting and limited fine-tuning | Full fine-tuning, quantisation, architecture access |
| **Ops burden** | Almost none | Substantial — this becomes your team's job |
| **Best for** | Getting to production fast, variable load | Strict data residency, very high steady volume, deep customisation |

**How to answer this in an interview:**

> "It depends on the constraints. If we're handling regulated data that can't leave our
> network, or we have very high steady volume where per-token pricing exceeds GPU costs,
> self-hosting an open-weights model makes sense. Otherwise a hosted API is almost always
> faster to ship and cheaper to operate, because you're not paying an engineering team to
> run inference infrastructure."

> 💡 **The best engineers design so the model is swappable.** Put your LLM calls behind
> one interface. Models improve every few months — if changing provider means rewriting
> your application, you've built yourself a cage.

---

## 🎤 Interview Questions

**Q1. What is an LLM?**
> A neural network trained to predict the next token given the preceding context. Modern
> chat models are then instruction-tuned and aligned so they follow instructions rather
> than merely continuing text.

**Q2. What is tokenization?**
> The process of converting text into tokens — the discrete units the model processes —
> which are then mapped to integer IDs and finally to embedding vectors.

**Q3. What are embeddings?**
> Numerical vector representations that capture semantic meaning, positioned so that
> similar meanings are close together in vector space.

**Q4. Why are embeddings important?**
> They enable semantic search, clustering, recommendation, deduplication, and RAG —
> matching on *meaning* rather than exact keywords.

**Q5. Why do LLMs hallucinate?**
> Because they generate statistically likely text rather than verifying facts. "I don't
> know" is rare in training data, so a model lacking knowledge still produces a fluent,
> confident continuation. Retrieval and validation reduce this; they don't eliminate it.

**Q6. If the context window is memory, why does the model forget?**
> It isn't memory. The model is stateless — the full conversation is resent on every
> request. Anything trimmed to fit the window is simply gone, and history management is
> the application's responsibility.

**Q7. Your RAG system is accurate but too expensive. What do you look at first?**
> Token usage, and specifically the retrieved context — usually the largest part of the
> prompt. I'd check how many chunks we retrieve and how large they are, add reranking so
> we can retrieve fewer but better chunks, cache repeated queries, and consider a smaller
> model for simpler requests.

---

## ❌ Common Beginner Mistakes

- ❌ Thinking LLMs search Google by default
- ❌ Believing embeddings and tokens are the same thing
- ❌ Assuming the context window is persistent memory
- ❌ Expecting the model to know when it is wrong
- ❌ Using a high temperature for factual or extraction tasks
- ❌ Assuming a bigger context window means all of it is used equally well
- ❌ Judging cost by request count instead of token count
- ❌ Hardcoding one provider throughout the codebase

---

## 🏆 Best Practices

- ✅ Count tokens **before** sending a request — for cost control and to avoid overflow errors
- ✅ Use **temperature 0** for anything factual or structured
- ✅ Treat every model output as **untrusted input** until validated
- ✅ Put your important context at the **start or end** of the prompt
- ✅ Manage conversation history explicitly — trim or summarise it
- ✅ Keep LLM calls behind **one swappable interface**
- ✅ Log tokens in, tokens out, latency, and cost for every single call

---

## 📝 Assignment

Answer these in your **own words** — writing them out is the point, because explaining
them out loud is exactly what an interview will require.

1. What is an LLM?
2. What is a token?
3. What is tokenization?
4. What is an embedding?
5. Why are embeddings used in RAG?
6. Why do hallucinations happen?
7. What is the context window?
8. Why does a production AI system often use RAG instead of relying only on the LLM?

**Bonus (practical):**

9. Install `tiktoken` and count the tokens in a paragraph of English and the same
   paragraph in Urdu. Which uses more tokens, and what does that mean for cost?
10. Explain why an LLM struggles to count the letters in "strawberry".

Save your answers in [`assignments/lesson-02/`](../../assignments/lesson-02/).

If you can answer questions 1–8 confidently and without notes, you already understand
concepts that many interview candidates cannot explain clearly.

---

## 📖 Resources

- [OpenAI Tokenizer (interactive)](https://platform.openai.com/tokenizer) — paste text and watch it split
- [tiktoken](https://github.com/openai/tiktoken) — the tokenizer library used in the code example
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — excellent visual primer, ideal preparation for Lesson 3
- [Lost in the Middle (Liu et al.)](https://arxiv.org/abs/2307.03172) — the long-context research referenced above
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) — compare embedding models

---

## 📊 Progress Tracker

**Module 1: Modern AI Engineering**

- ✅ Lesson 1 – Role of a Production AI Engineer
- ✅ Lesson 2 – How LLMs Actually Work
- ⬜ Lesson 3 – The Transformer Architecture

**Lessons remaining in Module 1:** 18
**Total bootcamp lessons remaining (approx.):** 118

---

## ➡️ What's Next – Lesson 3

**The Transformer Architecture**

We'll open the box labelled "Transformer Layers" above:

- Why transformers replaced RNNs and LSTMs
- Self-attention explained visually
- Query, Key, and Value (Q, K, V)
- Multi-head attention
- Positional encoding
- Feed-forward layers
- Residual connections
- Layer normalization
- Why transformers scale so well
- How GPT, Llama, Claude, and Qwen all build on this foundation

This is one of the most frequently tested topics in AI engineering interviews.

---

[⬅️ Lesson 1](lesson-01-what-does-a-production-ai-engineer-do.md) · [🏠 Course home](../../README.md)
