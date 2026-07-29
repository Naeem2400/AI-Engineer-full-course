# Lesson 8 – Embedding Models Masterclass (Industry Level)

> **Module:** 2 – Retrieval-Augmented Generation (RAG)
> **Level:** Intermediate → Expert
> **Estimated time:** 90 minutes

**Goal:** Learn how embedding models are built, how to choose the right one, and how
companies optimise retrieval quality in production RAG systems.

---

## 🚨 Why This Lesson Matters

Many beginners assume:

> ❌ *"Any embedding model will work."*

This is one of the most expensive mistakes in production AI. The wrong embedding model can
cost you a large fraction of your retrieval accuracy — even with an excellent LLM behind it.

Keep the division of labour clear:

> **The embedding model finds the information.**
> **The LLM writes the answer.**

If retrieval fails, the LLM never sees the correct information, and no amount of model
quality or prompt engineering recovers it. **Retrieval is the ceiling on your whole system.**

---

## 🗺️ Lesson Roadmap

1. What is an embedding model?
2. How embedding models are trained
3. Dense vs sparse embeddings
4. **Bi-encoder vs cross-encoder — and the arithmetic that forces the two-stage design**
5. The model families: SentenceTransformers, BGE, E5, Jina, Nomic, Voyage
6. Matryoshka embeddings
7. Multilingual embeddings — with honest caveats
8. Choosing a model
9. Benchmarking properly
10. Fine-tuning — and whether it's worth it
11. Interview questions

---

## 1. What Is an Embedding Model?

Lesson 4 established that text becomes a vector. This lesson answers: **who makes that
vector?**

```text
"Artificial Intelligence is changing healthcare."
                    │
                    ▼
            Embedding Model
                    │
                    ▼
        [0.14, -0.52, 0.91, ...]      768 numbers representing the MEANING
```

### In Simple Terms

Embeddings don't appear by themselves — an AI model generates them. Text goes in, a vector
comes out, and that vector represents what the text *means*.

---

## 2. How Are Embedding Models Trained?

```text
"The cat is sleeping."   ≈   "A kitten is resting."      → pull vectors TOGETHER
"The cat is sleeping."   ≠   "The stock market crashed." → push vectors APART
```

### Contrastive Learning

Modern embedding models are trained on triplets:

```text
Anchor:    "How to renew my passport?"
Positive:  "Passport renewal guide"        ← pull close
Negative:  "Pizza recipe"                  ← push far
```

### In Simple Terms

During training the model learns: same meaning → vectors close together; different meaning
→ vectors far apart. That is what makes semantic search possible.

### 🔑 The Detail That Separates Good Models From Great Ones: Hard Negatives

"Pizza recipe" is an **easy** negative — the model learns almost nothing from it, because
telling passports from pizza is trivial.

The valuable training signal comes from **hard negatives** — documents that look relevant
but aren't:

```text
Anchor:         "How to renew my passport?"
Positive:       "Passport renewal guide"
Hard negative:  "Passport application guide for first-time applicants"  ← so close!
```

*Renewal* versus *first-time application* is exactly the distinction a real user cares
about. Models trained on hard negatives learn fine-grained discrimination; models trained
on random negatives learn only coarse topic matching.

> 💡 This also explains a limitation from Lesson 4: contrastive training teaches
> **similarity**, never **logic**. That's why embeddings still fail on negation — nobody
> trained them to evaluate "with" versus "without".

---

## 3. Dense vs Sparse Embeddings

| | **Dense** | **Sparse** |
| --- | --- | --- |
| Looks like | `[0.12, -0.44, 0.91, ...]` | mostly zeros, a few weighted terms |
| Size | 384–3072 dims, all meaningful | vocabulary-sized (50k+), ~99% zeros |
| Captures | **Meaning** | **Exact terms** |
| Finds | "car" ↔ "automobile" | "PROTOCOL-CARD-2026-014" |
| Examples | BGE, E5, SentenceTransformers | BM25, **SPLADE** |

### In Simple Terms

Dense embeddings understand meaning. Sparse embeddings weight keywords. Production systems
usually combine both — hybrid search (Lesson 5).

### 🔑 SPLADE — The Interesting Middle Ground

BM25 is sparse but *not learned* — it just counts terms. **SPLADE** is a *learned* sparse
representation: a transformer predicts which vocabulary terms matter for a text, including
terms that never appear in it.

```text
Document: "myocardial infarction management"
SPLADE also activates:  heart(0.7)  cardiac(0.6)  attack(0.4)  treatment(0.5)
```

It performs **term expansion** — giving you some synonym-matching *while keeping the exact-term
precision of sparse retrieval*, and it works with existing inverted-index infrastructure.
Worth naming in an interview; most candidates only know dense vs BM25.

---

## 4. Bi-Encoder vs Cross-Encoder ⭐

The favourite enterprise interview topic — and the one where a concrete answer stands out.

### Bi-Encoder — Encode Separately

```text
Question ──► Encoder ──► vector ┐
                                ├──► cosine similarity
Document ──► Encoder ──► vector ┘
```

The critical property: **document vectors are computed once, offline, and reused forever.**
At query time you embed only the query, then do cheap vector arithmetic.

### Cross-Encoder — Encode Together

```text
[Question + Document] ──► one Transformer ──► relevance score (0–1)
```

The model sees both texts **simultaneously**, so attention (Lesson 3) can relate every
query word to every document word. Far more accurate.

**But there is no reusable index.** The score depends on the *pair*, so nothing can be
precomputed. Every candidate needs a full forward pass, at query time.

### 📊 The Arithmetic — Why This Isn't a Preference

For an international law firm: **15M documents ≈ 120M chunks**, 2-second budget.

```text
BI-ENCODER
  indexing (ONE TIME):  120,000,000 chunks x 2.0 ms  = 67 GPU-hours
  PER QUERY: embed query 2.0 ms + ANN search 15.0 ms = 17.0 ms   ← fits easily

CROSS-ENCODER OVER THE WHOLE CORPUS
  no reusable index: the query must be paired with EVERY chunk
  PER QUERY: 120,000,000 pairs x 8.0 ms = 960,000,000 ms
             = 267 hours per single query
             = 480,000x over the latency budget
  >>> INFEASIBLE. Not slow - impossible.

TWO-STAGE (what production actually does)
  rerank top 10  : 2.0 + 15.0 +   80.0 =    97.0 ms   OK
  rerank top 25  : 2.0 + 15.0 +  200.0 =   217.0 ms   OK
  rerank top 50  : 2.0 + 15.0 +  400.0 =   417.0 ms   OK
  rerank top 100 : 2.0 + 15.0 +  800.0 =   817.0 ms   OK

With a realistic 1200 ms reserved for LLM generation:
  available for reranking = 783 ms -> top 97 candidates
```

*(Per-item timings are representative GPU figures, not measurements from your hardware —
the ratios are the transferable part.)*

### What This Table Teaches

1. **Cross-encoders aren't "slower" — they're categorically infeasible at corpus scale.**
   267 hours per query. This is a difference in kind, not degree.
2. **The two-stage pattern isn't a clever optimisation — it's the only workable design.**
   A fast, imprecise retriever narrows millions to ~100; a slow, accurate reranker orders
   those 100.
3. **Your reranking budget is what's left after the LLM.** Reserve 1200 ms for generation
   and you can rerank ~97 candidates. That is how you actually pick "top-k for reranking" —
   from a latency budget, not from a blog post.

### The Production Pattern

```text
Bi-encoder ──► top 100 ──► Cross-encoder ──► top 10 ──► LLM
  (fast,                     (accurate,
   imprecise)                 expensive)
```

### In Simple Terms

Bi-encoder = fast search. Cross-encoder = final ranking. Production uses both. Lesson 9
covers rerankers in depth.

---

## 5. The Model Families

### SentenceTransformers

The standard open-source library (`all-MiniLM-L6-v2`, `mpnet-base-v2`, `msmarco-distilbert`).
Easy, free, well documented, and genuinely production-viable for many use cases.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
embedding = model.encode("What is Artificial Intelligence?", normalize_embeddings=True)
```

> ⚠️ It's a **library**, not a model family — it can load BGE, E5, and others too. People
> conflate these in interviews.

### BGE (BAAI)

`bge-small` / `base` / `large`, plus **BGE-M3** (multilingual, and notable for producing
dense, sparse, and multi-vector representations from one model). Strong retrieval quality,
widely used in enterprise RAG.

⚠️ English BGE models want a **query-side prefix**; passages get none.

### E5 (Microsoft)

`e5-small` / `base` / `large`, and `multilingual-e5`. **Requires prefixes on both sides:**

```python
query_vec = model.encode("query: How can I pay tax?")
doc_vec   = model.encode("passage: Tax payment instructions...")
```

**Interview question — why do E5 models use prefixes?**

> Because they were trained for *asymmetric* search, where a short question and a long
> document play different roles. The prefixes tell the model which role the text is
> playing. Omitting them doesn't error — it silently degrades retrieval.

### Jina, Nomic, Voyage

| Family | Known for |
| --- | --- |
| **Jina** | Long-document support, strong retrieval quality |
| **Nomic** | Open weights, local deployment, fully auditable pipelines |
| **Voyage** | Strong commercial retrieval quality, long context, domain-specific variants |

---

## 6. Matryoshka Embeddings

A genuinely useful recent development, and one that directly solves Lesson 4's storage
problem.

**Matryoshka Representation Learning (MRL)** trains a model so that the *most important
information sits in the earliest dimensions*. That means you can **truncate** the vector
and it still works:

```text
Full vector:      1536 dims  ──►  100% quality, 100% storage
Truncated:         768 dims  ──►  ~99% quality,  50% storage
Truncated:         512 dims  ──►  ~97% quality,  33% storage
Truncated:         256 dims  ──►  ~93% quality,  17% storage
```

*(Illustrative degradation — measure on your own data.)*

Normally, truncating an embedding destroys it. With MRL it degrades gracefully.

**Why this matters in production:**

- Cut your vector database RAM by 2–4× (recall the storage maths — 40M chunks at 1536 dims
  is ~230 GB of vectors alone)
- **Two-stage retrieval on one model:** search on truncated 256-dim vectors for speed, then
  rescore candidates using full 1536-dim vectors for accuracy
- Change your storage/quality trade-off **without re-embedding your corpus** — you keep
  the full vectors and simply index a prefix

> 🎤 Mentioning Matryoshka embeddings when asked "how would you reduce vector database
> costs?" is a strong signal you follow the field beyond tutorials.

---

## 7. Multilingual Embeddings

The goal: a user asks in Urdu, the document is in English, and retrieval still works.

```text
Query (Urdu):     پاکستان میں ٹیکس کیسے ادا کریں؟
Document (EN):    Tax payment instructions in Pakistan
                  ──► good multilingual models place these close together
```

Options: `multilingual-e5`, `BGE-M3`, Jina multilingual, Cohere multilingual.

### ⚠️ Three Honest Caveats

1. **Quality varies enormously by language pair.** English↔French/German/Spanish is
   usually strong. English↔Arabic/Urdu/Hindi is typically weaker — less training data and
   different scripts. **Never assume a "multilingual" label means uniform quality.** Test
   *your* languages.
2. **Multilingual models often underperform monolingual ones on English.** Capacity is
   shared across languages. If 95% of traffic is English, a strong English model plus
   translation for the rest may beat one multilingual model.
3. **Tokenisation cost differs sharply** (Lesson 2). Urdu and Arabic can consume 2–3× the
   tokens of equivalent English, which hits both your embedding costs and your chunk
   sizing — a "500-token chunk" holds far less Urdu content than English.

**A practical pattern for the law firm scenario:** evaluate one strong multilingual model
against per-language monolingual models routed by detected language. The second is more
infrastructure but often more accurate. Decide with measurements, not assumptions.

---

## 8. How to Choose an Embedding Model

| Criterion | Question to ask |
| --- | --- |
| **Accuracy** | Does it retrieve correctly *on my data*? |
| **Speed** | Latency at my request volume? |
| **Cost** | API per-token, or GPU hosting? |
| **Languages** | Which pairs, and tested how? |
| **Context length** | Can it embed my chunk sizes without truncation? |
| **Dimensions** | What does this cost in RAM at my scale? |
| **Prefixes** | What does the model card require? |
| **Licence** | Commercial use permitted? |
| **Stability** | Will this model still exist in two years? |

> 🔑 **The constraint people forget: max input length.** Text beyond it is **silently
> truncated** — no error, no warning. If your chunks are 1000 tokens and the model accepts
> 512, half of every chunk is being discarded and your retrieval quality is mysteriously
> poor. Check this *first* when debugging.

---

## 9. Benchmarking Properly

Companies don't guess. They measure.

| Metric | What it tells you |
| --- | --- |
| **Recall@k** | Is the correct document in the top-k at all? |
| **Precision@k** | How much of the top-k is relevant? |
| **MRR** | How high up is the *first* correct result? |
| **NDCG** | Ranking quality, weighted by position |
| **Latency (p50/p95/p99)** | Speed — and the tail is what users feel |
| **Cost per query** | Does this scale economically? |

An example comparison table:

| Model | Recall@10 | Avg latency |
| --- | ---: | ---: |
| MiniLM | 84% | 12 ms |
| BGE Large | 92% | 40 ms |
| E5 Large | 91% | 35 ms |

> ⚠️ **These numbers are illustrative, not measured** — they show the *shape* of the
> trade-off (bigger models score higher and cost more latency), and you should never
> quote figures like these as fact. Your corpus will produce different numbers.

### ⚠️ Why You Cannot Trust the Leaderboard Alone

MTEB is a useful starting filter, and a dangerous stopping point:

- **Benchmark overfitting.** Models are tuned to score well on public benchmarks. A high
  MTEB score partly measures MTEB skill.
- **Domain mismatch.** MTEB is general text. Your legal, medical, or internal corpus is not.
- **It ignores your constraints.** Latency, cost, licence, and language mix don't appear
  in the ranking.

**The correct workflow:**

```text
1. MTEB → shortlist 3-5 candidates matching your language and context-length needs
2. Build a golden set of 50-100 queries from YOUR data with known correct documents
3. Evaluate all candidates on it: recall@k, latency, cost
4. Pick using YOUR numbers
```

A golden set takes an afternoon and outlives every model you will ever swap in.

> 🔗 Distinguish this from Lesson 5's **index recall** (is ANN finding what brute force
> would?). Here we measure **retrieval quality** — is the right chunk findable at all?
> Perfect index recall over bad embeddings still returns the wrong documents.

---

## 10. Fine-Tuning Embedding Models

General-purpose embeddings sometimes aren't enough. A general model knows "heart" relates
to emotion and anatomy; a medical model knows it relates to cardiology, ECG, and
myocardial infarction.

Training data is query→relevant-document pairs from your domain:

```text
Query:     "How is diabetes treated?"
Positive:  "Diabetes treatment guidelines..."
Hard neg:  "Diabetes diagnosis criteria..."      ← the valuable signal (§2)
```

### ⚠️ Should You Actually Do This? Usually Not First

Fine-tuning is the most-suggested and least-often-correct answer to poor retrieval. Before
reaching for it, try these — all cheaper and usually more effective:

| Try first | Typical effort |
| --- | --- |
| Fix chunking and add headings (Lesson 7) | Hours |
| Verify prefixes and max input length | Minutes |
| Add hybrid search (Lesson 5) | Days |
| **Add a reranker (Lesson 9)** | Days — often the largest single gain |
| Fine-tune the **reranker** | Less data needed than embeddings |
| Fine-tune the **embedding model** | Weeks, and re-embed everything |

**Fine-tuning embeddings means re-embedding your entire corpus** every time you retrain,
plus building and maintaining a labelled dataset. On 120M chunks that is a serious
commitment.

**When it genuinely pays off:** highly specialised vocabulary that general models have
never seen (internal product codenames, rare legal or scientific terminology), *and* you
have thousands of real query→document pairs from actual usage logs, *and* you've already
exhausted the cheaper options above.

---

## 🏆 Production Best Practices

- ✅ Use the **same model and same prefix convention** for indexing and querying
- ✅ **Version your embeddings** — store model name and version with every vector
- ✅ Benchmark on **your own data**, not just leaderboards
- ✅ Pair a bi-encoder with a **cross-encoder reranker**
- ✅ Check **max input length** before choosing chunk size
- ✅ Consider **Matryoshka truncation** before buying more RAM
- ✅ Re-evaluate retrieval quality after every pipeline change

## ❌ Common Mistakes

- ❌ Different embedding models for documents and queries
- ❌ Choosing the largest model without checking latency
- ❌ Ignoring multilingual requirements — or assuming uniform multilingual quality
- ❌ Skipping retrieval evaluation entirely
- ❌ Forgetting to re-index after changing models
- ❌ **Missing model-specific prefixes** — silent degradation
- ❌ **Chunks exceeding max input length** — silent truncation
- ❌ Trusting MTEB rank as a decision
- ❌ Reaching for fine-tuning before fixing chunking and adding a reranker

> ⚠️ **A subtlety that sounds contradictory:** "use the same model for documents and
> queries" is correct — but asymmetric models use *different prefixes* for each side. Same
> model, different prefix. That is not an exception to the rule; it is the rule applied
> correctly.

---

## 🎤 Interview Questions

**Q1. What is an embedding model?**
> A neural model that converts text into dense vectors capturing semantic meaning, trained
> so that related texts land close together in vector space.

**Q2. Bi-encoder vs cross-encoder?**
> A bi-encoder embeds query and document independently, so document vectors are computed
> once offline and reused — enabling ANN search over millions. A cross-encoder processes
> the pair jointly for much better accuracy, but nothing can be precomputed, so it needs a
> forward pass per candidate at query time.

**Q3. Why not use a cross-encoder for everything?**
> It doesn't scale — with 120M chunks it's hundreds of hours per query, not a slow query
> but an impossible one. That's why production retrieves ~100 candidates with a bi-encoder
> and reranks only those.

**Q4. How do you decide how many candidates to rerank?**
> From the latency budget. Subtract LLM generation and retrieval from your target, divide
> the remainder by per-pair reranking cost. With a 2 s budget and 1.2 s for generation,
> that's roughly 100 candidates.

**Q5. Why must documents and queries use the same model?**
> Different models produce geometrically incompatible spaces. Mixing them doesn't error —
> it returns confident nonsense. Note that asymmetric models still use different *prefixes*
> for queries and passages.

**Q6. What is contrastive learning, and what makes it work well?**
> Training that pulls related pairs together and pushes unrelated pairs apart. Quality
> depends heavily on **hard negatives** — plausible-but-wrong documents — since random
> negatives only teach coarse topic matching.

**Q7. How would you cut vector database costs without hurting quality much?**
> Matryoshka truncation, scalar quantisation with full-precision rescoring, a smaller
> model if the golden set allows, or better chunking to reduce chunk count. Measure each
> against the golden set.

**Q8. Retrieval quality is poor. Walk me through debugging it.**
> Check prefixes match between index and query time; check chunks aren't exceeding max
> input length and being truncated; confirm the same model version indexed and queried;
> inspect actual retrieved chunks, not just final answers; test whether failures cluster on
> exact identifiers or negation, which points to hybrid search. Fine-tuning is last, not first.

---

## 📝 Assignment

You're building an AI assistant for an **international law firm**:

- 15 million legal documents
- English, French, German, and Arabic
- Responses in under 2 seconds
- High retrieval accuracy
- Cloud deployment

Design the retrieval pipeline and answer:

1. Which embedding model family would you evaluate first, and why?
2. Bi-encoder, cross-encoder, or both? Justify with the latency budget.
3. How would you benchmark different embedding models?
4. Which metrics would you monitor in production?
5. How would you support multiple languages?

**Harder:**

6. Compute the RAM for 15M documents at 1024 dims. How would you reduce it?
7. Arabic retrieval tests notably worse than French. Give three possible causes.
8. Given a 2-second budget with 1.2 s for LLM generation, how many candidates can you
   rerank? Show the calculation.

Save your answers in [`assignments/lesson-08/`](../../assignments/lesson-08/).

---

## 📖 Resources

- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- [Sentence-Transformers](https://www.sbert.net/)
- [BGE-M3](https://huggingface.co/BAAI/bge-m3) — multi-lingual, multi-granularity
- [E5 paper](https://arxiv.org/abs/2212.03533) — the prefix convention
- [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147)
- [SPLADE](https://arxiv.org/abs/2107.05720) — learned sparse retrieval

---

## 📊 Progress Tracker

**Module 2 – Retrieval-Augmented Generation (RAG)**

- ⬜ Lesson 6 – RAG from Beginner to Production *(not yet in this repo — see README)*
- ✅ Lesson 7 – Advanced Chunking Strategies
- ✅ Lesson 8 – Embedding Models Masterclass
- ⬜ Lesson 9 – Rerankers Deep Dive

**Lessons remaining in Module 2:** 17
**Overall bootcamp remaining (approx.):** 112 lessons

---

## ➡️ What's Next – Lesson 9

**Rerankers Deep Dive (The Secret Behind High-Quality RAG)**

Why vector search alone is insufficient · cross-encoder rerankers in depth · Cohere-style
reranking · BGE and Jina rerankers · multi-stage retrieval · context compression ·
latency vs accuracy · production architectures · hands-on Python · system design.

This lesson covers the technique that most often separates an "okay" RAG system from an
excellent one — and section 4 has already shown you why it exists.

---

[⬅️ Lesson 7](lesson-07-advanced-chunking-strategies.md) · [🏠 Course home](../../README.md)
