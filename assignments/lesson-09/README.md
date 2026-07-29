# Assignment – Lesson 9

## Exercise 1 — Pipeline Design

Sketch a complete production pipeline using:

- User query
- Embedding model
- Qdrant
- BGE Reranker
- LLM (GPT / Claude / Llama)
- Final response

For each stage, label:

- What it optimises — **recall** or **precision**?
- What you would measure there
- What happens if that stage fails

---

## Exercise 2 — The Reranker's Effect

You retrieved 20 documents, but only 4 are truly relevant.

Explain how reranking changes the final prompt, and give **three distinct reasons** the
answer improves. ("Fewer tokens" is only one of them.)

---

## Homework

1. Draw the complete production RAG pipeline from memory.
2. Compare **bi-encoder vs cross-encoder** in your own words — include *why* the
   cross-encoder is more accurate, not just that it is.
3. Research **BGE Reranker** and note:
   - What problem it solves
   - Its advantages
   - Its limitations

---

## Harder

4. Your **recall@50 is 0.72** and **precision@5 after reranking is 0.95**. Where is the
   problem, and what do you fix? (Careful — the high precision is a distraction.)
5. Reranking added 400 ms and your p95 latency breached the SLA. Give **three** options
   and the trade-off of each.
6. Why might reranking *reduce hallucinations*, beyond simply sending fewer tokens?

---

## How to Submit

Add your answers to this folder as either:

- `solution.md` — written answers plus a text diagram, or
- `solution.ipynb` — a notebook, if you want to implement a reranking comparison

---

## Self-Check

You have genuinely understood this lesson if you can explain, without notes:

- [ ] Why vector search ranks all chunks about the same topic almost identically
- [ ] Why a bi-encoder *cannot* answer "does this chunk answer this question?"
- [ ] **That retrieval sets the ceiling and reranking only approaches it**
- [ ] How to derive retrieval depth k from a recall curve and a latency budget
- [ ] Why a cross-encoder is more accurate, in terms of attention (Lesson 3)
- [ ] Two situations where a reranker makes things worse
- [ ] Why chunk *ordering* matters after reranking, not just chunk selection

> 💡 **The debugging habit worth building:** when RAG returns a wrong answer, don't ask
> "is my reranker good enough?" first. Ask **"was the correct chunk in the candidate pool
> at all?"** Log your retrieved candidate IDs. Recall failures and precision failures look
> identical to the user and need completely different fixes.
