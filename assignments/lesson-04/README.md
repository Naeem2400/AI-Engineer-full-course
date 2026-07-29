# Assignment – Lesson 4

## Part 1 — Diagram

Draw the RAG pipeline and explain each step in your own words:

```text
User Question ──► Embedding Model ──► Vector Database ──► Top-k Chunks ──► LLM ──► Answer
```

For each arrow, answer: *what goes in, what comes out, and what could go wrong here?*

---

## Part 2 — Core Questions

1. What is an embedding?
2. Why do embeddings enable semantic search?
3. Why is cosine similarity commonly used?
4. Why does RAG use embeddings instead of keyword matching?
5. What's the difference between a token and an embedding?

---

## Part 3 — Harder (Production Reality)

6. Your team wants to switch from `all-MiniLM-L6-v2` (384 dims) to a 1536-dimension
   model. What has to happen, and what does it cost?
7. Give two concrete queries where embeddings will **fail** and keyword search will
   **succeed**. Explain why.
8. You have 5 million chunks. Compare the RAM needed for a 384-dim model vs a 1536-dim
   model. Show your working.

---

## Part 4 — Practical

9. Run the numpy script from section 9 of the lesson. Confirm that cosine, dot product,
   and Euclidean distance produce the **same ranking** on normalised vectors. Then remove
   the normalisation and observe the rankings diverge. Why does dot product change?

10. Install `sentence-transformers` and embed these three sentences:

    - *"How can I lose weight?"*
    - *"Best diet plan for fat loss"*
    - *"How do I fix my car?"*

    Compute the similarity matrix. Do the results match your expectation? Which pair
    scores highest, and do they share any words?

---

## How to Submit

Add your answers to this folder as either:

- `solution.md` — written answers plus a text diagram, or
- `solution.ipynb` — a notebook, if you want to include the numpy and embedding experiments

---

## Self-Check

You have genuinely understood this lesson if you can explain, without notes:

- [ ] Why "dense" matters — how embeddings differ from TF-IDF / one-hot vectors
- [ ] That embedding models are trained by **contrastive learning**, and why that means
      they don't understand negation
- [ ] Why cosine, dot product, and Euclidean give identical rankings on normalised vectors
- [ ] What a **query prefix** is, and why forgetting one degrades quality *silently*
- [ ] Why you cannot swap embedding models without re-embedding everything
- [ ] Three concrete things embeddings are bad at, and the mitigation for each
- [ ] Why chunking strategy affects quality more than model choice

If any of these are shaky, re-read that section before Lesson 5.

> 💡 The two mistakes that cost real teams real weeks are in section 10: **missing query
> prefixes** and **mixed embedding models**. Neither one throws an error. Both just make
> your retrieval quietly worse. Learn to check for them first when RAG "just isn't working".
