# Lesson 5 – Vector Databases & ANN Search (The Heart of Production RAG)

> **Module:** 1 – Modern AI Engineering
> **Level:** Intermediate → Expert
> **Estimated time:** 90 minutes

**Goal:** Understand how companies like Microsoft, Thomson Reuters, Amazon, Google, and
OpenAI store and search millions — or billions — of embeddings in milliseconds.

---

## 🎯 Why This Lesson Matters

Imagine a company holding:

- 📄 50 million legal documents
- 📄 100 million tax documents
- 📄 20 million medical records

A user asks:

> *"What are the UK VAT rules for digital services?"*

The answer must arrive in **under 2 seconds**. Lesson 4 gave you the vectors. This lesson
is about how you search 500 million of them without waiting an hour.

---

## 🗺️ Lesson Roadmap

1. Why SQL isn't enough
2. What is a vector database?
3. Exact search vs approximate search
4. ANN — Approximate Nearest Neighbour
5. HNSW, and the parameters you'll actually tune
6. Top-k retrieval
7. Metadata filtering — and the pre/post-filter trap
8. Hybrid search and Reciprocal Rank Fusion
9. Quantisation — fitting billions of vectors in RAM
10. Measuring recall (you must)
11. Popular vector databases
12. Production architecture
13. When you *don't* need a vector database

---

## 1. Why Can't We Use MySQL or PostgreSQL Alone?

A normal table:

| ID | Name |
| --- | --- |
| 1 | Apple |
| 2 | Orange |

```sql
SELECT * FROM fruits WHERE name = 'Apple';
```

This works because we're matching an **exact value**.

Now search by **meaning**:

```text
User asks:        "Heart attack treatment"
Database holds:   "Myocardial infarction therapy"
```

SQL sees two unrelated strings. A vector database sees two nearly identical vectors.

### In Simple Terms

SQL compares exact text. If a user writes "Car" and the database says "Automobile", SQL
finds nothing. A vector database finds it, because it compares meaning rather than
spelling.

### The Deeper Reason

You *can* store vectors in a plain SQL column and compute similarity in a query. The
problem is that it's a **full table scan** — every row, every time. Vector databases exist
because they add a specialised **index** that avoids looking at most of the data. That
index is the entire point of this lesson.

---

## 2. What Is a Vector Database?

A vector database stores **embeddings** plus **metadata**, and indexes them for similarity
search.

```text
"The company policy on parental leave..."
                │
                ├──►  vector:   [0.34, -0.92, 1.76, ...]
                │
                ├──►  payload:  { "department": "HR",
                │                 "country": "UK",
                │                 "year": 2026 }
                │
                └──►  text:     the original chunk (for citations)
```

**Store the original text alongside the vector.** You cannot reconstruct it from the
embedding, and you need it both to build the prompt and to show the user a citation.

---

## 3. Exact Search vs Approximate Search

With 500 million embeddings, exact search means:

```text
Query ──► compare with vector 1 ──► vector 2 ──► ... ──► vector 500,000,000
```

This is **exact search** (also called flat, or brute-force, or k-NN). It is perfectly
accurate and completely impractical at scale.

### In Simple Terms

Comparing against every vector isn't impossible — it's just far too slow. Production
systems don't do it.

> ⚠️ **But don't dismiss it.** Under roughly 50,000 vectors, brute-force search with numpy
> takes a few milliseconds and gives **100% recall** with zero infrastructure. Reaching
> for a vector database on a 5,000-chunk corpus is over-engineering. See section 13.

---

## 4. Approximate Nearest Neighbour (ANN)

ANN searches only the most promising **regions** of the vector space instead of everything.

> A library has 10 million books. You don't scan every shelf — you go to the right
> section first.

### Why "Approximate"?

It returns the *best practical* matches, not a mathematical guarantee of the true nearest
neighbours. You might get 9 of the true top 10.

**This trade-off is almost always worth it:**

| | Exact | ANN |
| --- | --- | --- |
| Recall | 100% | 95–99% (tunable) |
| Speed at 1M vectors | ~seconds | ~milliseconds |
| Memory | Vectors only | Vectors + index overhead |

Losing the 10th-best chunk rarely changes the LLM's answer. Waiting 4 seconds definitely
changes the user's experience.

---

## 5. HNSW — Hierarchical Navigable Small World

The dominant ANN algorithm, and a favourite interview topic.

### The Motorway Analogy

To cross a country you don't take local streets the whole way:

```text
Motorway     ──►   Main Road   ──►   Local Street   ──►   Destination
(few, fast,        (moderate)        (many, precise)
 long hops)
```

HNSW builds exactly this, as layers of a graph:

```text
Layer 2   ●─────────────────────●          few nodes, long jumps
          │                     │
Layer 1   ●────────●────────────●          more nodes, medium hops
          │        │            │
Layer 0   ●──●──●──●──●──●──●──●──●        every vector, short hops
```

Search enters at the top, greedily hops toward the query, drops a layer, refines, and
repeats. Most of the data is never touched.

### 🔧 The Three Parameters You Will Actually Tune

This is where interviews go past the analogy:

| Parameter | Controls | Raise it → | Lower it → |
| --- | --- | --- | --- |
| **M** | Edges per node | Better recall, more memory | Less memory, worse recall |
| **ef_construction** | Candidates explored while *building* | Better index, slower build | Fast build, weaker index |
| **ef_search** | Candidates explored while *querying* | Better recall, slower query | Faster query, lower recall |

**The one that matters day to day is `ef_search`** — it's a *query-time* knob, so you can
tune latency against recall **without rebuilding the index**. Start around 64–128 and
measure.

Typical starting points: `M = 16`, `ef_construction = 200`, `ef_search = 64`.

### Trade-offs Worth Knowing

- ✅ Very fast, high recall, scales well
- ⚠️ **Memory hungry** — the graph lives in RAM alongside the vectors
- ⚠️ **Slow to build** — indexing millions of vectors takes real time
- ⚠️ **Deletes are awkward** — usually a tombstone plus periodic rebuild

Used by Qdrant, Weaviate, Milvus, pgvector, Elasticsearch, and most others.

### Alternatives You Should Recognise

| Index | Idea | Best for |
| --- | --- | --- |
| **Flat** | Compare everything | < ~50k vectors, exact results |
| **IVF** | Cluster vectors, search nearest clusters | Large static datasets |
| **HNSW** | Layered graph | The general-purpose default |
| **IVF-PQ** | Clustering + compression | Billion-scale, memory constrained |

---

## 6. Top-K Retrieval

```text
Query: "Annual tax return"

Document        Score
Tax Guide        0.96   ✅
VAT Manual       0.91   ✅   k = 2
HR Policy        0.18
Travel Rules     0.12
```

Only the top-k chunks go to the LLM.

### In Simple Terms

Top-k means we don't send all documents — only the most relevant ones. If k = 5, the LLM
receives just the 5 best chunks.

### 🔑 Choosing k Is a Real Engineering Decision

| k too small | k too large |
| --- | --- |
| Misses the answer entirely | Irrelevant chunks distract the model |
| Cheap, fast | Expensive — recall Lesson 3's O(n²) attention |
| | "Lost in the middle" (Lesson 2) buries the good chunk |

**More context is not better context.** A common production pattern is to retrieve
generously and then narrow:

```text
Vector search (k = 50)  ──►  Reranker  ──►  top 5  ──►  LLM
```

The retriever is fast and imprecise; the reranker is slow and accurate. Running the
reranker on 50 candidates rather than 5 million is what makes the combination affordable.

---

## 7. Metadata Filtering — And the Trap

Store structured fields alongside vectors:

```json
{ "country": "UK", "department": "Finance", "year": 2026, "security_level": "internal" }
```

Filtering improves speed, accuracy, **and security**. The `user_id` filter that stopped
the cross-tenant leak in the Lesson 1 assignment was metadata filtering.

### ⚠️ Pre-Filter vs Post-Filter — The Production Gotcha

This distinction breaks real systems and almost no tutorial mentions it.

**Post-filtering** — search first, then filter:

```text
ANN search (top 10)  ──►  keep only country = "UK"  ──►  maybe 2 results left ❌
```

If UK documents are rare, your top 10 might contain **zero** of them. You asked for 10
results and got 2. Or none.

**Pre-filtering** — restrict the candidate set, then search:

```text
country = "UK"  ──►  search only those vectors  ──►  10 genuine UK results ✅
```

Correct, but naive pre-filtering can break the HNSW graph: if you remove most nodes, the
remaining graph may be disconnected and the search gets stuck.

**How good databases solve it:** Qdrant's *filterable HNSW* applies the filter **during**
graph traversal, keeping the graph navigable. Weaviate and Milvus have equivalents.

> 🎤 **Interview answer:** *"With a highly selective filter I'd want pre-filtering, because
> post-filtering can return fewer than k results. I'd check whether the database supports
> filtering during graph traversal — Qdrant calls it filterable HNSW — since naive
> pre-filtering can disconnect the HNSW graph."* That answer is well above the median.

---

## 8. Hybrid Search

Lesson 4 listed what embeddings are bad at: exact identifiers, rare jargon, negation.
Hybrid search is the fix.

```text
        Query
          │
    ┌─────┴─────┐
    ▼           ▼
BM25 keyword   Vector search
    │           │
    └─────┬─────┘
          ▼
    Fuse the rankings (RRF)
          ▼
       Reranker
          ▼
         LLM
```

### Why Both?

| Query | Winner | Why |
| --- | --- | --- |
| `VAT-2026-Section-15` | **BM25** | Exact token; embeddings blur near-identical IDs |
| `digital tax rules` | **Vector** | Paraphrase; no exact keyword to match |
| `myocardial infarction` in a corpus saying "heart attack" | **Vector** | Pure synonym |
| A rare internal codename | **BM25** | Never appeared in embedding training data |

Real queries are a mix, so production runs both.

### 🔑 How Results Are Actually Merged: Reciprocal Rank Fusion

Your draft said "merge results" — here's the actual mechanism. You **can't** compare a
BM25 score to a cosine score; they're different scales. RRF sidesteps this by using only
**rank position**:

```text
                    1
RRF(d) =  Σ    ────────────           k ≈ 60 by convention
        systems  k + rank(d)
```

```python
def reciprocal_rank_fusion(rankings, k=60):
    scores = {}
    for ranking in rankings:                      # one list per retriever
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)
```

Elegant properties: no score normalisation needed, no tuning of a blend weight, and
documents that *both* retrievers rank highly rise to the top.

---

## 9. Quantisation — Fitting Billions of Vectors in RAM

From Lesson 4: 1M × 1536 dims × 4 bytes ≈ 6.1 GB. At 500 million vectors that's **3 TB**.
Quantisation compresses vectors at a small accuracy cost.

| Method | Compression | Recall impact |
| --- | --- | --- |
| **Scalar (float32 → int8)** | 4× | Minimal — usually the easy win |
| **Product Quantisation (PQ)** | 10–50× | Noticeable; needs a rescoring pass |
| **Binary (1 bit per dim)** | 32× | Large — only with rescoring on full vectors |

**The standard production pattern** is two-stage retrieval:

```text
Search compressed vectors (fast, approximate)  ──►  top 100
                                                     │
Rescore those 100 against full-precision vectors  ──►  top 10
```

You get most of the memory saving and nearly all of the accuracy, because you only need
full precision on a handful of candidates.

---

## 10. Measuring Recall — Do This, Don't Skip It

ANN gives up exactness. **How much?** If you can't answer with a number, you don't know
whether your retrieval is broken.

**Recall@k** = of the true top-k nearest neighbours, what fraction did ANN return?

```python
def recall_at_k(approx_ids, exact_ids, k):
    return len(set(approx_ids[:k]) & set(exact_ids[:k])) / k
```

Compute exact results with brute force on a **sample** of queries (a few hundred is
plenty), then compare. Target ≥ 0.95 for most applications.

> 💡 There are two different things to measure, and people conflate them:
> **Index recall** — is ANN finding what brute force would? (this section)
> **Retrieval quality** — is the right chunk in the top-k at all? (Lesson 4's golden set)
> An index at 0.99 recall retrieving the wrong chunks means your *embeddings or chunking*
> are wrong, not your index.

---

## 11. Popular Vector Databases

| Database | Model | Strengths | Consider when |
| --- | --- | --- | --- |
| **Qdrant** | Open source, self-host or cloud | Fast, excellent filtering, great Python SDK | Default choice for self-hosted RAG |
| **Weaviate** | Open source / cloud | Built-in hybrid search, modules, GraphQL | You want batteries included |
| **Pinecone** | Managed only | Zero ops, autoscaling | You don't want to run infrastructure |
| **Milvus** | Open source | Built for billion-scale | Very large datasets |
| **pgvector** | PostgreSQL extension | SQL + relational data + vectors in one DB | You already run Postgres |
| **Elasticsearch / OpenSearch** | Search engine | Mature BM25 + vectors together | Hybrid search matters most |

### 💡 Honest Advice on Choosing

For most teams, **pgvector** or **Qdrant** is the right answer.

- Already running Postgres with under a few million vectors? **pgvector.** One less system
  to operate, transactional consistency with your relational data, and your existing
  backups already cover it.
- Need heavy filtering, higher scale, or vector-native features? **Qdrant.**

Choose a purpose-built database when you have a *measured* reason. "We might scale to a
billion vectors" is not a measured reason on day one.

---

## 12. Production Architecture

```text
                    User
                      │
                      ▼
              FastAPI Backend
                      │
                      ▼
              Authentication            ← who is this, what may they see?
                      │
                      ▼
           Query Embedding Model        ← SAME model + prefix as indexing (Lesson 4!)
                      │
        ┌─────────────┴─────────────┐
        ▼                           ▼
   BM25 Keyword               Vector DB (ANN)
     Search                  + metadata pre-filter
        │                           │
        └─────────────┬─────────────┘
                      ▼
              RRF Fusion (§8)
                      │
                      ▼
              Reranker (top 50 → 5)
                      │
                      ▼
              Prompt Builder
                      │
                      ▼
                     LLM
                      │
                      ▼
             Safety Validation         ← grounding check (Lesson 1)
                      │
                      ▼
            Response + Citations
```

Compare this with the Lesson 1 diagram. Same skeleton — you now understand every box.

---

## 13. When You *Don't* Need a Vector Database

Every tutorial tells you to install one. Sometimes that's wrong:

| Situation | Better choice |
| --- | --- |
| < ~10,000 chunks | numpy brute force in memory — milliseconds, 100% recall, zero ops |
| < ~1M chunks, already on Postgres | pgvector |
| Small corpus that fits in the context window | Skip retrieval entirely — just send it |
| Exact-match lookups only | Plain SQL with an index |

> 🎤 **A senior answer to "which vector database would you use?"** — *"How many vectors,
> what's the latency budget, and what infrastructure do we already run? Under a few
> million with Postgres in place, I'd use pgvector and avoid a new system. Beyond that, or
> with heavy filtering, Qdrant."* Asking those questions scores higher than naming a
> product.

---

## 14. Python Example

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

client = QdrantClient("localhost", port=6333)

results = client.query_points(
    collection_name="documents",
    query=query_embedding,
    limit=5,
    query_filter=Filter(                       # pre-filter, applied during traversal
        must=[
            FieldCondition(key="country", match=MatchValue(value="UK")),
            FieldCondition(key="department", match=MatchValue(value="Finance")),
        ]
    ),
    search_params={"hnsw_ef": 128},            # the recall/latency knob (§5)
    with_payload=True,                          # return text + metadata for citations
).points

for point in results:
    print(point.score, point.payload["text"][:80])
```

Note three things: the filter is passed **into** the search (not applied afterwards),
`hnsw_ef` is tunable per query, and `with_payload=True` returns the text you need for
citations.

---

## 🎤 Interview Questions

**Q1. Why do we need a vector database?**
> Relational databases index exact values, so semantic search means a full scan and a
> similarity computation over every row. Vector databases add ANN indexes that avoid
> examining most of the data, turning a linear scan into a sub-linear search.

**Q2. What is ANN?**
> A family of algorithms that find highly similar vectors far faster than exhaustive
> comparison, trading a small, measurable amount of recall for a large speed gain.

**Q3. What is HNSW?**
> A layered graph index. Upper layers have few nodes and long edges for coarse navigation;
> lower layers are dense for precision. Search descends the layers, touching a tiny
> fraction of the data.

**Q4. Which HNSW parameters would you tune, and when?**
> `M` and `ef_construction` at build time for index quality versus memory and build time;
> `ef_search` at query time to trade latency against recall. `ef_search` is the practical
> knob because it needs no rebuild.

**Q5. What's the difference between pre-filtering and post-filtering?**
> Post-filtering searches then discards non-matching results, which can return fewer than
> k results when the filter is selective. Pre-filtering restricts the candidate set first,
> which is correct but can disconnect the HNSW graph unless the database supports
> filtering during traversal.

**Q6. What is hybrid search, and how are the results merged?**
> Running keyword search and vector search together. Because BM25 and cosine scores aren't
> comparable, results are usually merged with Reciprocal Rank Fusion, which scores by rank
> position rather than raw score.

**Q7. How do you know your ANN index isn't losing results?**
> Measure recall@k against brute-force ground truth on a sample of queries, and raise
> `ef_search` until recall meets target. Track it as a regression test.

**Q8. Your RAG latency is 4 seconds. Where do you look?**
> Break the latency down by stage before changing anything: query embedding, vector
> search, reranking, and LLM generation. Usually the LLM dominates, in which case streaming
> and a smaller model help more than index tuning. If search dominates, look at
> `ef_search`, k, filter strategy, and whether the index fits in RAM.

---

## ❌ Common Beginner Mistakes

- ❌ Storing raw text without embeddings
- ❌ Sending the whole document collection to the LLM
- ❌ Assuming SQL can replace a vector database for semantic search
- ❌ Using an excessively large k — cost up, quality often down
- ❌ **Never measuring recall** — you cannot tune what you don't measure
- ❌ **Post-filtering when you needed pre-filtering** — silently returns too few results
- ❌ Adding a vector database for a 5,000-chunk corpus
- ❌ Forgetting the index must fit in RAM, and budgeting for vectors only
- ❌ Using different embedding models at index and query time (Lesson 4)

---

## 📝 Assignment

Design a RAG system for a hospital with:

- 5 million medical documents
- Doctors searching in natural language
- Metadata: Department, Country, Language, Publication Date

Draw the architecture and answer:

1. Why would you choose a vector database?
2. Which retrieval method would you use?
3. Would you use hybrid search? Why?
4. How would metadata filtering improve performance?
5. Why is ANN preferable to exact search here?

**Harder:**

6. Compute the RAM needed for 5M vectors at 1024 dimensions. Does it fit on one machine?
7. A doctor filters to `Department = Cardiology`, which is 2% of documents. Explain what
   goes wrong with post-filtering.
8. Give one query where hybrid search returns the right document and pure vector search
   fails.

👉 **A complete worked solution with runnable benchmarks is provided in
[`assignments/lesson-05/solution.ipynb`](../../assignments/lesson-05/solution.ipynb).**

---

## 📖 Resources

- [HNSW paper (Malkov & Yashunin)](https://arxiv.org/abs/1603.09320)
- [Qdrant documentation](https://qdrant.tech/documentation/) — filterable HNSW explained
- [pgvector](https://github.com/pgvector/pgvector)
- [BM25 explained](https://en.wikipedia.org/wiki/Okapi_BM25)
- [Reciprocal Rank Fusion (Cormack et al.)](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
- [ANN Benchmarks](https://ann-benchmarks.com/) — index algorithms compared empirically

---

## 📊 Progress Tracker

**Module 1: Modern AI Engineering**

- ✅ Lesson 1 – What a Production AI Engineer Does
- ✅ Lesson 2 – How LLMs Work
- ✅ Lesson 3 – Transformer Architecture
- ✅ Lesson 4 – Embeddings Deep Dive
- ✅ Lesson 5 – Vector Databases & ANN Search
- ⬜ Lesson 6 – RAG from Beginner to Production

**Lessons remaining in Module 1:** 15
**Total bootcamp lessons remaining (approx.):** 115

---

## ➡️ What's Next – Lesson 6

**Retrieval-Augmented Generation (RAG) from Beginner to Production**

Why RAG exists · the full pipeline · ingestion · chunking strategies · embedding
generation · retrieval · reranking · prompt construction · generation · evaluation ·
common production failures · enterprise architecture · hands-on implementation.

Arguably the most important topic for a GenAI Engineer, and asked in almost every modern
AI interview.

---

[⬅️ Lesson 4](lesson-04-embeddings-deep-dive.md) · [🏠 Course home](../../README.md)
