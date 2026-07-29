# Assignment – Lesson 2

## Task

Answer these questions in your **own words**. Do not copy the lesson text — the whole
point is to be able to explain these concepts out loud, which is exactly what an
interview requires.

### Core Questions

1. What is an LLM?
2. What is a token?
3. What is tokenization?
4. What is an embedding?
5. Why are embeddings used in RAG?
6. Why do hallucinations happen?
7. What is the context window?
8. Why does a production AI system often use RAG instead of relying only on the LLM?

### Bonus (Practical)

9. Install `tiktoken` and count the tokens in a paragraph of English text, then the same
   paragraph translated into Urdu. Which uses more tokens? What does that mean for the
   cost of a multilingual product?
10. Explain why an LLM struggles to count the letters in the word "strawberry".

---

## How to Submit

Add your answers to this folder as either:

- `solution.md` — plain text answers, or
- `solution.ipynb` — a notebook, if you want to include the `tiktoken` experiments

---

## Self-Check

You have genuinely understood this lesson if you can explain, without notes:

- [ ] Why "the model predicts the next token" explains hallucinations
- [ ] The difference between a **token embedding** (inside the LLM) and a
      **text embedding** (a separate model you call for search)
- [ ] Why the context window is **not** memory, and what actually happens when
      ChatGPT "remembers" earlier messages
- [ ] Why output tokens are slower and more expensive than input tokens
- [ ] When you would choose a closed API model over open weights, and why

If any of these are shaky, re-read that section before moving to Lesson 3 — the
Transformer lesson assumes all of it.
