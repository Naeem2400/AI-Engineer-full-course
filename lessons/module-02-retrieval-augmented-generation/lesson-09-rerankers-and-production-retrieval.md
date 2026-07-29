# Lesson 9 – Rerankers & Production Retrieval Pipelines

> **Module:** 2 – Retrieval-Augmented Generation (RAG)
> **Level:** Intermediate → Expert
> **Estimated time:** 75–90 minutes

**Goal:** Understand why production RAG systems never rely on vector search alone, and how
rerankers dramatically improve answer quality.

Lesson 8 proved *why* rerankers must exist (a cross-encoder over a full corpus needs 267
hours per query). This lesson is about *using* one well.

---

## 🎯 What You'll Learn

- ✅ Why vector search isn't enough — **demonstrated with measurements**
- ✅ What rerankers are and how they score
- ✅ **The ceiling: what reranking can and cannot fix**
- ✅ How to choose retrieval depth from data
- ✅ BGE, Cohere, Jina, Voyage rerankers
- ✅ Hybrid search + reranking
- ✅ Context compression and chunk ordering
- ✅ How to prove a reranker actually helped

---

## 1. The Production Architecture

```text
User Question
      │
      ▼
Embedding Model
      │
      ▼
Vector Database  ──►  Retrieve Top 50
      │
      ▼
   Reranker      ──►  Select Best 5
      │
      ▼
Prompt Builder
      │
      ▼
    LLM  ──►  Answer
```

**Vector search does not feed the LLM directly.** A reranker sits between them.

### In Simple Terms

Think of Google. It first narrows billions of pages down to a few hundred candidates, then
a second, smarter system decides which of those is genuinely most relevant. That second
system is the **reranker**.

---

## 2. Why Vector Search Isn't Enough

A user asks:

> *"What side effects does Liprixa cause?"*

Every chunk in the drug's documentation is *about Liprixa*. So they all embed similarly —
description, dosage, storage, clinical trials, side effects. **Topic similarity cannot
distinguish between them.**

### The Core Problem

A bi-encoder embeds each chunk **alone**, before your question exists. It answers:

> *"What is this chunk about?"*

It cannot answer:

> *"Does this chunk answer **this specific question**?"*

That second question requires seeing the query and the document **together** — which is
exactly what a cross-encoder does.

---

## 3. 📊 Measured: What Reranking Actually Does

A 10-chunk pharmaceutical corpus. Only **C4** and **C7** actually answer the question.

```text
QUERY: 'What side effects does Liprixa cause?'
Chunks that actually answer it: ['C4', 'C7']
====================================================================

STAGE 1 - BI-ENCODER (vector search), top 5
--------------------------------------------------------------------
   1. C2   0.946            Liprixa dosage information: 20mg once daily, t
   2. C5   0.940            Liprixa should be stored below 25 degrees Cels
   3. C6   0.937            Liprixa is contraindicated in patients with se
   4. C1   0.931            Liprixa is prescribed to help lower cholestero
   5. C10  0.930            The Liprixa patient information leaflet was up

STAGE 2 - CROSS-ENCODER RERANK of top 8, top 5
--------------------------------------------------------------------
   1. C4   0.990  RELEVANT  Patients taking Liprixa experienced nausea, vo
   2. C7   0.990  RELEVANT  Adverse reactions to Liprixa include dizziness
   3. C2   0.200            Liprixa dosage information: 20mg once daily, t
   4. C5   0.200            Liprixa should be stored below 25 degrees Cels
   5. C6   0.200            Liprixa is contraindicated in patients with se
```

### Read the Numbers

```text
  precision@1   bi-encoder 0.00   ->   reranked 1.00
  precision@2   bi-encoder 0.00   ->   reranked 1.00
  precision@3   bi-encoder 0.00   ->   reranked 0.67
```

**Vector search put zero relevant chunks in its top 5.** Not because it's broken — because
every chunk genuinely is about Liprixa, and all scored between 0.930 and 0.946. Those tiny
gaps are noise, not relevance.

The reranker read the *question* alongside each chunk and separated them decisively:
**0.990 for the side-effect chunks, 0.200 for everything else.** That's the difference
between "same topic" and "answers the question".

### And the Cost Effect

```text
CONTEXT SENT TO THE LLM
  without reranking (top 8) :  128 tokens
  with reranking (top 3)     :   46 tokens
  reduction                  : 64%
```

Fewer tokens means lower cost, lower latency, **and better answers** — because the model
isn't sifting six irrelevant passages to find one useful one (recall "lost in the middle"
from Lesson 2).

> ⚠️ **Honest disclosure:** both encoders here are *simulated* — the bi-encoder scores
> topic overlap, the cross-encoder scores query-intent against chunk content. No trained
> model was available offline. The **failure mode and the fix are real**; the exact numbers
> are a teaching aid. Run this with a real `bge-reranker` on your corpus to get numbers you
> can quote.

---

## 4. ⚠️ The Ceiling: What Reranking Cannot Fix

**This is the most important idea in the lesson, and the one most tutorials omit.**

A reranker can only reorder what retrieval handed it. If the right chunk isn't in the
candidate pool, no reranker on earth can surface it.

```text
THE CEILING: a reranker can only reorder what retrieval gave it
--------------------------------------------------------------------
  retrieve top 1  -> relevant chunks in pool: NONE     | reranked #1: C2 <-- WRONG
  retrieve top 2  -> relevant chunks in pool: NONE     | reranked #1: C2 <-- WRONG
  retrieve top 3  -> relevant chunks in pool: NONE     | reranked #1: C2 <-- WRONG
  retrieve top 5  -> relevant chunks in pool: NONE     | reranked #1: C2 <-- WRONG
  retrieve top 8  -> relevant chunks in pool: ['C4', 'C7'] | reranked #1: C4 OK
  retrieve top 10 -> relevant chunks in pool: ['C4', 'C7'] | reranked #1: C4 OK
```

Retrieve 5 and the system fails completely — *with a reranker attached*. Retrieve 8 and it
works perfectly. **The reranker didn't change. The retrieval depth did.**

### State It Precisely

| Stage | Improves | Fixed by |
| --- | --- | --- |
| **Retrieval** (bi-encoder, depth k) | **Recall** — is the right chunk *available*? | Better embeddings, chunking, hybrid search, larger k |
| **Reranking** (cross-encoder) | **Precision** — is the right chunk *on top*? | Better reranker |

> 🔑 **Retrieval sets the ceiling. Reranking approaches it.**
>
> If your recall@50 is 0.80, then 20% of questions are unanswerable no matter how good
> your reranker is. Adding a better reranker to a low-recall retriever is polishing a
> window in a house with no door.

**This reframes debugging.** When RAG gives a wrong answer, the first question isn't "is my
reranker good enough?" — it's **"was the correct chunk in the candidate pool at all?"** Log
your retrieved candidate IDs and check. The two failures need completely different fixes.

---

## 5. How to Choose Retrieval Depth (k)

Most teams pick k=50 because a blog said so. Derive it instead — it's a 20-minute measurement.

### Step 1 — Measure Your Retriever's Recall Curve

Using your golden set (Lesson 7 §9):

```python
for k in [5, 10, 20, 50, 100, 200]:
    recall = fraction_of_golden_queries_where_correct_chunk_in_top_k(k)
    print(k, recall)
```

You'll see something like:

```text
k=5    0.61
k=10   0.74
k=20   0.86
k=50   0.94      ← the knee
k=100  0.96
k=200  0.965     ← diminishing returns
```

### Step 2 — Pick the Knee, Then Check the Latency Budget

Beyond the knee you pay linearly more reranking latency for almost no recall. From Lesson
8's calculation: with a 2-second budget and ~1.2 s reserved for the LLM, you can rerank
roughly **100 candidates**.

```text
recall knee (k=50)  ≤  latency ceiling (k≈100)   ✅ use k=50
```

If the knee sat *above* your latency ceiling, you'd have a genuine problem — and the fix
would be improving **retrieval** (better embeddings, hybrid search, better chunking), not
buying a faster reranker.

---

## 6. Bi-Encoder vs Cross-Encoder (Recap)

| | Bi-Encoder | Cross-Encoder |
| --- | --- | --- |
| Input | Query and document **separately** | Query + document **together** |
| Precompute | ✅ Document vectors, once, offline | ❌ Nothing — score depends on the pair |
| Speed | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| Accuracy | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Scales to | Millions of documents | ~100 candidates per query |
| Role | Retrieval | Final ranking |

The cross-encoder is more accurate for a structural reason from Lesson 3: when both texts
share one input sequence, **attention can relate every query token to every document
token**. A bi-encoder has already compressed the document to a fixed vector before your
question exists — that information is simply gone.

> 🔗 Lesson 8 §4 has the full arithmetic on why this forces the two-stage design.

---

## 7. Popular Rerankers

| Reranker | Type | Notes |
| --- | --- | --- |
| **BGE Reranker** (BAAI) | Open source | The usual default. `base` / `large` / `v2-m3` (multilingual). Runs locally — no data leaves your network |
| **Cohere Rerank** | Cloud API | Very easy to integrate, strong quality, multilingual. Per-call cost, and your documents leave your infrastructure |
| **Jina Reranker** | Open + API | Fast, good multilingual support |
| **Voyage Rerank** | Cloud API | Strong quality, domain-specific variants |

### Choosing

- **Regulated data (medical, legal, financial)?** Self-host BGE. Cloud reranking means
  sending document *content* — not just embeddings — to a third party. This is a compliance
  question before it is a quality one.
- **Prototyping or moderate volume?** Cohere Rerank is the fastest path to a working system.
- **Multilingual?** Test `bge-reranker-v2-m3` and Jina; verify on **your** language pairs
  (Lesson 8 §7 — multilingual quality varies sharply by pair).

```python
# Self-hosted, sentence-transformers
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("BAAI/bge-reranker-base")
pairs = [(query, chunk.text) for chunk in candidates]      # ~50 pairs
scores = reranker.predict(pairs)

ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])[:5]
```

Note the shape: **one call, batched over all candidates.** Scoring candidates in a Python
loop one at a time is a common and costly mistake — batch them.

---

## 8. Hybrid Search + Reranking

The full retrieval stack:

```text
        Query
          │
    ┌─────┴─────┐
    ▼           ▼
  BM25      Vector search
 (top 50)     (top 50)
    │           │
    └─────┬─────┘
          ▼
    RRF fusion (Lesson 5 §8)
          ▼
      ~80 unique candidates
          ▼
       Reranker
          ▼
        top 5
          ▼
         LLM
```

Each stage does one job: **hybrid search maximises recall** (get the right chunk into the
pool); **reranking maximises precision** (get it to the top).

This also resolves the honest problem we hit in the Lesson 5 notebook — RRF rewards
consensus and can rank a correct single-retriever hit below a mediocre both-retriever hit.
**The reranker fixes exactly that**, because it judges relevance directly rather than by
rank position.

---

## 9. Two Refinements Worth Knowing

### Chunk Ordering — Free Quality

You have relevance scores. Use them for **placement**, not just selection. From Lesson 2,
models attend most reliably to the start and end of context, and can miss the middle.

```text
❌ chunks in retrieval order          ✅ best-first (or best at both ends)
```

Reordering costs nothing and you already have the scores.

### Context Compression

Even a relevant 800-token chunk may contain two useful sentences. **Extractive compression**
selects only the sentences that matter before building the prompt:

```text
Reranked chunks (5 x 800 tokens = 4000)
        ↓ extract relevant sentences
Compressed context (~800 tokens)
```

Worth it when context is your cost driver. The trade-off: you risk cutting context the LLM
needed, so measure answer quality — not just token savings — before adopting it.

---

## 10. ⚠️ When Rerankers Hurt

Rerankers are not free, and treating them as a magic upgrade causes real problems:

| Cost | Detail |
| --- | --- |
| **Latency** | Reranking 50 candidates adds meaningful time to every request |
| **Another model to operate** | Hosting, versioning, monitoring, GPU capacity |
| **Another dependency** | A cloud reranker outage takes your RAG down — build a fallback that skips reranking rather than failing |
| **Out-of-domain weakness** | A general reranker on highly specialised text can rank *worse* than your retriever |
| **Data egress** | Cloud reranking sends document content to a third party |

> ⚠️ **Always A/B against no reranking.** It usually helps substantially. "Usually" is not
> "always", and on your corpus you should have the number.

---

## 11. Proving the Reranker Helped

Never ship a reranker on faith. Measure on your golden set, before and after:

| Metric | What it tells you |
| --- | --- |
| **Recall@k (retriever)** | Your ceiling — unchanged by reranking |
| **Precision@5 (after rerank)** | Is the context clean? |
| **MRR** | How high is the *first* correct chunk? |
| **NDCG@10** | Overall ranking quality, position-weighted |
| **Tokens per request** | The cost saving (§3 measured 64%) |
| **p95 latency** | What reranking cost you |
| **End-to-end answer accuracy** | The only metric that actually matters |

The last row is the point. Better retrieval metrics that don't improve answers mean you
optimised the wrong thing.

---

## 12. A Realistic Production Stack

- **Embeddings:** BGE, E5, Voyage, or an API model
- **Vector DB:** Qdrant, Weaviate, Pinecone, Milvus, pgvector
- **Keyword:** Elasticsearch / OpenSearch (BM25)
- **Fusion:** RRF
- **Reranker:** BGE Reranker, Cohere Rerank, Jina
- **LLM:** Claude, GPT, Llama, Mistral
- **Framework:** FastAPI + LangChain/LangGraph, or custom orchestration
- **Observability:** Langfuse, Phoenix, or OpenTelemetry

> 💡 On frameworks: LangChain is excellent for getting to a prototype quickly. Many teams
> later replace parts of it with direct calls, because debugging production retrieval
> through several abstraction layers is painful. Either choice is defensible — knowing
> *why* you chose it is what matters in an interview.

---

## 13. Diagnosing a Degraded RAG System

A realistic symptom set — rising token usage, more validation failures, wrong chunk IDs
appearing in citations:

**Reranking is a strong candidate fix**, because retrieving 50 and passing all 50 to the
LLM inflates tokens and invites the model to cite the wrong passage. Narrowing to the best
5 reduces both.

But diagnose before prescribing:

1. **Are the correct chunks in the candidate pool?** (§4 — recall problem, not precision)
2. **Did chunking change?** Wrong chunk IDs can mean re-indexing altered boundaries
3. **Do chunk IDs still map to the right text?** Stale index versus fresh vectors
4. **Is the embedding model or its prefix convention the same as at index time?** (Lesson 8)
5. **Then** add reranking, and measure the delta

> The order matters. "Add a reranker" as a reflex is a guess; "check whether the right
> chunk was even retrieved, then add a reranker" is engineering.

---

## 🎤 Interview Questions

**Q1. Why not send all retrieved documents to the LLM?**
> Token cost and latency rise, and answer quality often falls — irrelevant context invites
> the model to use the wrong passage, and material buried mid-context gets missed. A
> reranker keeps only what's relevant.

**Q2. Difference between embeddings and rerankers?**
> Embeddings are a bi-encoder producing reusable vectors for fast retrieval over millions
> of documents. A reranker is a cross-encoder scoring query and document jointly — far more
> accurate, but it must run per candidate at query time, so it only sees ~100.

**Q3. Can a reranker fix poor retrieval?**
> No. It only reorders what retrieval returned. If the correct chunk isn't in the top-k
> pool, the reranker can never surface it. Retrieval sets the ceiling; reranking approaches
> it. Low recall is fixed with better embeddings, chunking, hybrid search, or larger k.

**Q4. How do you choose retrieval depth k?**
> Measure the retriever's recall@k curve on a golden set, take the knee where recall
> plateaus, then confirm it fits the latency budget for reranking. If the knee exceeds what
> you can afford to rerank, improve retrieval rather than reranking more.

**Q5. Cross-encoders are more accurate — why not use one for retrieval?**
> Nothing can be precomputed, so it needs a forward pass per document at query time. Over
> 100M chunks that's hundreds of hours per query. It's infeasible, not merely slow.

**Q6. When might a reranker make things worse?**
> Out-of-domain data where a general reranker misjudges specialised text, latency-critical
> paths, or when it becomes a single point of failure. Always A/B against no reranking.

**Q7. How do you prove a reranker helped?**
> Golden-set metrics before and after — precision@5, MRR, NDCG, tokens per request, p95
> latency — and most importantly end-to-end answer accuracy. Retrieval metrics that don't
> move answer quality mean the wrong thing was optimised.

---

## 📝 Assignment

### Exercise 1 — Pipeline Design

Sketch a complete pipeline using: user query → embedding model → Qdrant → BGE Reranker →
LLM → response. Label what each stage optimises (recall or precision) and where you'd
measure.

### Exercise 2 — The Reranker's Effect

You retrieved 20 documents; only 4 are truly relevant. Explain how reranking changes the
final prompt, and give **three** distinct reasons the answer improves.

### Homework

1. Draw the complete production RAG pipeline from memory.
2. Compare bi-encoder vs cross-encoder in your own words — include *why* the cross-encoder
   is more accurate, not just that it is.
3. Research **BGE Reranker**: what problem it solves, its advantages, and its limitations.

### Harder

4. Your recall@50 is 0.72 and precision@5 after reranking is 0.95. Where is the problem,
   and what do you fix?
5. Reranking added 400 ms and your p95 breached the SLA. Give three options and their
   trade-offs.
6. Why might reranking *reduce* hallucinations, beyond simply sending fewer tokens?

Save your answers in [`assignments/lesson-09/`](../../assignments/lesson-09/).

---

## 📖 Resources

- [BGE Reranker](https://huggingface.co/BAAI/bge-reranker-large)
- [Cohere Rerank](https://docs.cohere.com/docs/rerank-overview)
- [Sentence-Transformers CrossEncoder](https://www.sbert.net/examples/applications/cross-encoder/README.html)
- [Lost in the Middle](https://arxiv.org/abs/2307.03172) — why chunk ordering matters
- [Pinecone — Rerankers and Two-Stage Retrieval](https://www.pinecone.io/learn/series/rag/rerankers/)

---

## 📊 Progress Tracker

**Module 2 – Retrieval-Augmented Generation (RAG)**

- ⬜ Lesson 6 – RAG from Beginner to Production *(not yet in this repo — see README)*
- ✅ Lesson 7 – Advanced Chunking Strategies
- ✅ Lesson 8 – Embedding Models Masterclass
- ✅ Lesson 9 – Rerankers & Production Retrieval Pipelines
- ⬜ Lesson 10 – Advanced Prompt Engineering

**Completed:** 9 lessons · **Remaining:** ~21

---

## ➡️ What's Next – Lesson 10

**Advanced Prompt Engineering for Production LLM Systems**

Prompt templates · system prompts · tool calling · structured outputs · prompt versioning ·
and the techniques used in enterprise GenAI applications.

---

[⬅️ Lesson 8](lesson-08-embedding-models-masterclass.md) · [🏠 Course home](../../README.md)
