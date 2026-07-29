# Assignment – Lesson 8

## Scenario

You're building an AI assistant for an **international law firm**:

| Requirement | Value |
| --- | --- |
| Documents | 15 million legal documents |
| Languages | English, French, German, Arabic |
| Latency | Under 2 seconds end to end |
| Accuracy | High retrieval quality required |
| Deployment | Cloud |

Design the retrieval pipeline, then answer:

### Core Questions

1. Which embedding model family would you evaluate first, and why?
2. Bi-encoder, cross-encoder, or both? **Justify using the latency budget.**
3. How would you benchmark different embedding models?
4. Which metrics would you monitor in production?
5. How would you support multiple languages?

### Harder

6. Compute the RAM needed for 15M documents at 1024 dimensions. State your
   chunks-per-document assumption. How would you reduce it?
7. Arabic retrieval tests notably worse than French. Give **three** possible causes and
   how you'd distinguish them.
8. Given a 2-second budget with 1.2 s reserved for LLM generation, how many candidates
   can you rerank? Show the calculation.

---

## How to Submit

Add your answers to this folder as either:

- `solution.md` — written answers plus an architecture diagram, or
- `solution.ipynb` — a notebook, if you want to include the capacity and latency maths

---

## Self-Check

You have genuinely understood this lesson if you can explain, without notes:

- [ ] Why a cross-encoder cannot replace a bi-encoder — in **orders of magnitude**, not
      vague terms
- [ ] Why document vectors can be precomputed but cross-encoder scores cannot
- [ ] How to derive your reranking candidate count from a latency budget
- [ ] What **hard negatives** are and why they matter more than easy ones
- [ ] What SPLADE gives you that neither BM25 nor dense embeddings do
- [ ] How Matryoshka embeddings cut storage without re-embedding the corpus
- [ ] Why "same model for docs and queries" and "different prefixes for each" are both true
- [ ] Why MTEB rank is a shortlist, never a decision
- [ ] Why fine-tuning embeddings is usually the **last** fix to try, not the first

> 💡 **The two silent failures to check first when retrieval is bad:** missing model
> prefixes, and chunks exceeding the model's max input length. Neither throws an error.
> Both quietly wreck quality.
