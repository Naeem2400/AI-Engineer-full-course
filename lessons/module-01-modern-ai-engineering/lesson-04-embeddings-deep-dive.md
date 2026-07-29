# Lesson 4 – Embeddings Deep Dive (The Foundation of RAG)

> **Module:** 1 – Modern AI Engineering
> **Level:** Intermediate → Expert
> **Estimated time:** 75–90 minutes

**Goal:** Understand embeddings from beginner to interview level, and know how semantic
search actually works.

---

## 🎯 Why This Lesson Matters

If I had to choose **one topic** that separates a beginner from a professional GenAI
Engineer, it would be **embeddings**.

Almost every enterprise AI application depends on them:

ChatGPT Memory · Enterprise Search · Legal AI · Medical AI · Recommendation Systems ·
AI Agents · RAG · Document Search

**If you don't understand embeddings, you cannot build production RAG systems.**

---

## 🗺️ Lesson Roadmap

1. Why can't AI understand text directly?
2. What is an embedding?
3. Semantic space — the map analogy
4. How big are embeddings?
5. How embedding models are trained
6. Sentence embeddings
7. Semantic search
8. Measuring similarity: cosine, dot product, Euclidean
9. **The metric truth most tutorials get wrong**
10. Query prefixes — the #1 production bug
11. Chunking: where RAG quality is won or lost
12. What embeddings are *bad* at
13. Choosing an embedding model
14. Interview questions

---

## 1. Why Can't AI Understand Text?

Take the word:

```text
Apple
```

A human immediately knows this could mean 🍎 the fruit or 🍏 the company — and works out
which from context.

A computer sees only characters:

```text
A   p   p   l   e
```

Letters carry no meaning. Comparing strings tells you whether two things are *spelled*
alike, never whether they *mean* alike.

So we need a different representation.

### In Simple Terms

A computer only sees text as characters — it doesn't understand meaning. To teach AI what
words actually mean, we convert text into numbers. Those numbers are called embeddings.

---

## 2. What Is an Embedding?

> **An embedding is a dense numerical vector that represents the meaning of text.**

Simplified example:

```text
Doctor    ──►  [ 0.24, -0.88,  0.51,  1.20 ]
Hospital  ──►  [ 0.21, -0.90,  0.48,  1.15 ]   ← very close to Doctor
Teacher   ──►  [-0.70,  0.30, -0.45,  0.12 ]   ← far away
```

Doctor and Hospital end up mathematically close. Teacher doesn't.

**"Dense" matters.** Contrast with the old sparse approach:

| | Sparse (one-hot / TF-IDF) | Dense (embeddings) |
| --- | --- | --- |
| **Size** | Vocabulary-sized (50,000+) | 384–3,072 |
| **Contents** | Mostly zeros | Every dimension carries signal |
| **"Doctor" vs "Physician"** | Completely unrelated | Nearly identical |
| **Captures meaning?** | No — only word identity | Yes |

### In Simple Terms

An embedding is a list of numbers — but not random numbers. They represent the meaning of
the text. Words with similar meanings get vectors that sit close together.

---

## 3. Semantic Space — The Map Analogy

Think of Google Maps. Every restaurant has a latitude and longitude, and nearby
restaurants have nearby coordinates.

Embeddings work the same way — except the coordinates represent **semantic location**
instead of physical location:

```text
        ▲
        │        ⚕️ Doctor
        │      Physician
        │       Surgeon                    🍕 Pizza
        │       Hospital                    Burger
        │                                   Pasta
        │
        │              ⚽ Football
        │                 Cricket
        │                 Tennis
        └──────────────────────────────────────────►
```

Related concepts form **clusters**. This is called **semantic space**, and everything in
RAG is just navigating it.

> 💡 A real embedding space has hundreds or thousands of dimensions — impossible to
> visualise directly. Engineers use UMAP or t-SNE to project it down to 2D for inspection.
> Worth doing once on your own data: clusters that *should* be separate but overlap
> usually mean your chunking is wrong.

---

## 4. How Big Are Embeddings?

| Model | Dimensions |
| --- | ---: |
| all-MiniLM-L6-v2 | 384 |
| BGE Large | 1024 |
| E5 Large | 1024 |
| OpenAI-style embeddings | 1536+ |
| Some research models | 4096+ |

More dimensions allow richer representations, but they are **not automatically better** —
and they are never free.

### 💰 The Storage Maths Nobody Shows You

```text
1 million chunks × 1536 dimensions × 4 bytes (float32)  =  ~6.1 GB
1 million chunks ×  384 dimensions × 4 bytes            =  ~1.5 GB
```

That's **4× the RAM** for the larger model — and vector indexes are usually held in
memory. On a 10-million-chunk corpus this is the difference between one server and four.

**Practical guidance:** benchmark the small model first. If a 384-dimension model gives
you 95% of the retrieval quality at 25% of the cost and latency, that is usually the
correct engineering decision.

---

## 5. How Are Embedding Models Trained?

This is a common interview question and most candidates can't answer it.

Embedding models are trained with **contrastive learning**. The model sees triplets:

```text
Anchor:    "How do I reset my password?"
Positive:  "Steps to recover your account login"      ← should be CLOSE
Negative:  "Our refund policy for annual plans"       ← should be FAR
```

The loss function pulls the anchor and positive together in vector space, and pushes the
anchor and negative apart. Repeat over millions of pairs and the geometry of the space
organises itself by meaning.

> 🔑 **The insight that explains everything else:** embedding models are trained to place
> *similar things nearby*. They are **not** trained to understand logic, negation, or
> arithmetic. Section 12 shows exactly where that bites.

### Where Do the Pairs Come From?

Naturally occurring pairs at internet scale: question/answer pairs from forums, title/body
pairs from articles, citations, translation pairs. **Hard negatives** — examples that look
similar but aren't relevant — matter most for quality, because they force the model to
learn fine distinctions.

---

## 6. Sentence Embeddings

We usually embed **whole sentences or chunks**, not individual words.

```text
Sentence A:  "How can I lose weight?"
Sentence B:  "Best diet plan for fat loss."
```

**Zero words in common.** Yet the meanings are nearly identical, and the embedding model
places them close together.

### In Simple Terms

AI doesn't look at exact words — it looks at meaning. That's why "lose weight" and "fat
loss" are treated as similar.

### ⚠️ Token Embeddings vs. Text Embeddings (Recap from Lesson 2)

| | **Token embeddings** | **Text/sentence embeddings** |
| --- | --- | --- |
| Lives | *Inside* the LLM's first layer | A *separate* model you call |
| Represents | One token | A whole sentence or chunk |
| Used for | Nothing directly — internal | Search, RAG, clustering |

When an AI engineer says "embeddings", they mean the **second** kind.

---

## 7. Semantic Search

### Traditional Keyword Search

```text
User  ──►  Keyword Matching  ──►  Results
```

Search for `Car`, document says `Automobile` → **no result**. The words don't match, so
the search fails, even though the meaning is identical.

### Semantic Search

```text
User Question  ──►  Embedding  ──►  Vector Database  ──►  Similar Vectors  ──►  Documents
```

Now `Car` finds `Automobile`, because their vectors are close.

> ⚠️ **This is not a strict upgrade.** Keyword search still wins on exact identifiers —
> error codes, invoice numbers, product SKUs, surnames. Section 12 covers why, and the
> answer is usually to run **both** (hybrid search, Lesson 5).

---

## 8. Measuring Similarity

Given two vectors, how do we know if they're similar? Three metrics dominate.

### Cosine Similarity

Measures the **angle** between two vectors, ignoring their length.

```text
     ↗ A          A and B point the same way  ──►  similar
    ↗ B

     ↗ A          A and C point apart         ──►  dissimilar
      ↘ C
```

| Score | Meaning |
| --- | --- |
| `1.0` | Identical direction |
| `0.0` | Unrelated (perpendicular) |
| `-1.0` | Opposite direction |

```python
def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    return dot / (norm_a * norm_b)
```

### Dot Product

Considers **both direction and magnitude**. Cheaper to compute — no square roots — which
is why vector databases like it.

### Euclidean Distance (L2)

The straight-line distance between two points. **Smaller = more similar.** Note this is a
*distance*, not a similarity — the sort order is reversed.

### In Simple Terms

Cosine similarity compares the angle between two vectors. If both point in the same
direction, similarity is high. That's why it's so popular in vector databases.

---

## 9. 🔑 The Metric Truth Most Tutorials Get Wrong

Tutorials present cosine, dot product, and Euclidean as three meaningful choices. Here is
what actually happens in production:

> **If your vectors are normalised to unit length, all three metrics produce the
> IDENTICAL ranking.**

And **most modern embedding models output normalised vectors already.**

### Proof, Not Assertion

```python
import numpy as np
rng = np.random.default_rng(42)

query = rng.normal(size=384)
docs  = rng.normal(size=(10, 384))

def normalise(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)

qn, dn = normalise(query), normalise(docs)

cosine = dn @ qn
dot    = dn @ qn
euclid = np.linalg.norm(dn - qn, axis=1)

print(list(np.argsort(-cosine)))   # best-first by cosine
print(list(np.argsort(-dot)))      # best-first by dot
print(list(np.argsort(euclid)))    # best-first by distance (ascending)
```

**Real output:**

```text
ranking by cosine (desc) : [8, 2, 9, 7, 0, 1, 3, 6, 4, 5]
ranking by dot    (desc) : [8, 2, 9, 7, 0, 1, 3, 6, 4, 5]
ranking by euclid (asc)  : [8, 2, 9, 7, 0, 1, 3, 6, 4, 5]

cosine == dot ranking?     True
cosine == euclid ranking?  True

identity check ||a-b||² = 2 - 2·cos :
  max abs difference = 8.88e-16          ← floating-point noise, i.e. exactly equal
```

### Why This Is True

For unit vectors, the norms are 1, so:

```text
cosine(a,b) = a · b                     ← cosine IS the dot product
‖a − b‖² = 2 − 2·cos(a,b)               ← distance is a decreasing function of cosine
```

Euclidean distance shrinks monotonically as cosine grows, so ordering by one is
identical to ordering by the other.

### And When Vectors Are *Not* Normalised

The same script with varied magnitudes:

```text
ranking by cosine : [8, 2, 9, 7, 0, 1, 3, 6, 4, 5]
ranking by dot    : [9, 8, 2, 0, 7, 6, 1, 3, 4, 5]
same ranking?      False
```

They now **disagree**. Dot product rewards long vectors, so a merely-okay match with a
large magnitude can outrank a great match with a small one.

### What This Means for You

| Situation | What to do |
| --- | --- |
| Vectors normalised (most modern models) | Pick whichever your DB optimises — usually **dot product**, it's fastest |
| Vectors not normalised | Use **cosine**, or normalise first. Dot product will bias toward long vectors |
| Model documentation specifies a metric | **Follow it** — it was trained and evaluated with that metric |

> 🎤 **Interview gold:** *"For normalised embeddings, cosine and dot product give the same
> ranking, and dot product is cheaper because it skips the norm computation. The choice
> only matters when vectors aren't normalised."* Very few candidates know this.

---

## 10. ⚠️ Query Prefixes — The #1 Production Bug

This single mistake silently destroys retrieval quality, and it is everywhere.

Several popular models — **E5, BGE, and others** — are trained for **asymmetric search**,
where the question and the document play different roles. They require **prefixes**:

```python
# E5 models
query_vec = model.encode("query: How much maternity leave is allowed?")
doc_vec   = model.encode("passage: Employees are entitled to 26 weeks...")

# BGE models (English) — prefix the QUERY only
query_vec = model.encode("Represent this sentence for searching relevant passages: " + question)
doc_vec   = model.encode(document)      # no prefix
```

**Forget the prefix and nothing errors.** No exception, no warning. You get vectors, you
get results, and your retrieval quality is just quietly worse — sometimes dramatically so.
Teams have lost weeks to this.

### Rules to Live By

1. **Read your embedding model's model card before writing any code.** Prefix requirements
   are documented there and nowhere else.
2. **Use the same prefix convention at indexing time and at query time.** A mismatch is as
   bad as no prefix.
3. **Symmetric models** (like `all-MiniLM-L6-v2`) need no prefixes — question and document
   are embedded identically.

### 🚨 And Never Mix Embedding Models

Vectors from different models are **mathematically incompatible**. They live in different
spaces with different geometry. Searching a BGE index with an OpenAI query vector doesn't
error — it returns confident nonsense.

**Consequence:** changing your embedding model means **re-embedding your entire corpus**.
On millions of documents that's real time and real money. Choose deliberately, and store
the model name and version alongside your vectors so you always know what generated them.

---

## 11. Chunking — Where RAG Quality Is Won or Lost

Before you can embed documents, you must split them. This decision affects retrieval
quality more than your choice of embedding model, and most tutorials give it one line.

### The Core Tension

| Chunks too small | Chunks too large |
| --- | --- |
| Context is cut off mid-idea | Meaning gets diluted across topics |
| "26 weeks" without "maternity leave" | One vector must represent 5 unrelated things |
| High precision, poor completeness | Retrieval gets vague and imprecise |
| More chunks = more storage | Wastes context window and money |

### Sensible Starting Points

| Content type | Chunk size | Overlap |
| --- | --- | --- |
| General prose / articles | 500–1000 tokens | 10–15% |
| Dense legal / medical text | 300–500 tokens | 15–20% |
| Code | By function or class | Keep signatures intact |
| FAQs | One Q&A pair per chunk | None needed |

### Why Overlap Matters

Without it, a sentence split across a boundary becomes unfindable by either chunk:

```text
No overlap:
  Chunk 1: "...employees are entitled to"
  Chunk 2: "26 weeks of maternity leave..."
           ↑ neither chunk answers "how much maternity leave?"

With overlap:
  Chunk 1: "...employees are entitled to 26 weeks of maternity leave"
  Chunk 2: "entitled to 26 weeks of maternity leave. Applications must..."
           ↑ both are findable
```

### Better Than Fixed-Size: Respect Structure

Split on **semantic boundaries** — headings, paragraphs, sections — before falling back to
size limits. A chunk that begins mid-sentence embeds badly. Keep tables and lists intact.

> 💡 **Practical trick:** prepend the document title and section heading to every chunk
> before embedding. A chunk reading *"26 weeks are granted"* is ambiguous alone; as
> *"HR Policy > Parental Leave: 26 weeks are granted"* it embeds far more accurately.

---

## 12. ⚠️ What Embeddings Are *Bad* At

Every tutorial sells embeddings. Almost none tell you where they fail — and knowing this
is what makes you useful in a design review.

### 1. Negation

```text
"Contracts WITH an indemnity clause"
"Contracts WITHOUT an indemnity clause"
```

These embed **very close together**, despite meaning opposite things. Recall section 5:
the model was trained to place similar-looking text nearby, never to evaluate logic.

**In a legal or medical product this is a serious risk.** Mitigate with reranking,
metadata filters, or having the LLM verify the retrieved chunk actually matches the
condition asked about.

### 2. Exact Identifiers

```text
"Error code E4021"      vs.   "Error code E4022"
"Invoice INV-2024-8871"
```

Nearly identical vectors, completely different referents. Embeddings capture *the gist*,
and the gist of two error codes is the same.

**This is the single strongest argument for hybrid search** — BM25 keyword matching
handles exact tokens precisely, embeddings handle meaning. Production systems run both.
(Lesson 5.)

### 3. Numbers and Comparisons

*"Revenue above £5 million"* will happily retrieve a chunk saying *"revenue of
£2 million"*. Embeddings don't do arithmetic. Numeric filtering belongs in **metadata**,
not in the vector.

### 4. Rare Domain Jargon

If your internal product codenames never appeared in training data, the model has no
meaningful representation for them. Options: fine-tune the embedding model on your domain,
or lean on hybrid search.

### 5. Very Long Text

Every embedding model has a maximum input length, and text beyond it is **silently
truncated** — you'll never see an error. Embedding a 50-page document into one vector
produces a blurry average that matches nothing well. This is precisely why we chunk.

---

## 13. How RAG Uses Embeddings — End to End

```text
INDEXING (once per document)
────────────────────────────
  PDF 1: Tax Laws        PDF 2: Employment Rules      PDF 3: Healthcare Policies
                                    │
                                    ▼
                    Split into chunks (§11)
                                    │
                                    ▼
                       Embedding Model  ──►  vectors
                                    │
                                    ▼
              Vector Database (Qdrant / Weaviate / Pinecone)
                     stored: vector + text + metadata


QUERYING (every question)
─────────────────────────
  "How much maternity leave is allowed?"
                    │
                    ▼
          Embedding Model   ← the SAME model, with the SAME prefix convention
                    │
                    ▼
          Vector Search (top-k)
                    │
                    ▼
          3–5 most relevant chunks     ← not the whole corpus
                    │
                    ▼
                  LLM
                    │
                    ▼
          Answer + citations
```

**The critical point:** the LLM never sees all your company documents. It receives only
the few most relevant chunks. That's why RAG is fast, affordable, and accurate — and, per
Lesson 3, why it avoids the O(n²) cost of an enormous prompt.

---

## 14. Python Example

The standard library for open-source embeddings:

```python
# pip install sentence-transformers
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "Artificial Intelligence",
    "Machine Learning",
    "Football",
]

embeddings = model.encode(sentences, normalize_embeddings=True)
print(embeddings.shape)      # (3, 384)

# With normalised vectors, cosine similarity is just a dot product (§9)
similarity = embeddings @ embeddings.T
print(similarity.round(3))
```

You would expect "Artificial Intelligence" and "Machine Learning" to score notably higher
against each other than either does against "Football".

> ⚠️ Note `normalize_embeddings=True`. Set it, and everything in section 9 applies: your
> similarity computation collapses to a single matrix multiply.

---

## 15. Choosing an Embedding Model

### Popular Models (2026)

**Open source (Hugging Face)**
- `all-MiniLM-L6-v2` — 384 dims, fast, a great baseline
- `BAAI/bge-large-en` — strong English quality
- `BAAI/bge-m3` — multilingual, multi-granularity
- `intfloat/e5-large-v2` — strong retrieval, needs `query:`/`passage:` prefixes
- `jina-embeddings-v3` — long input support

**Cloud APIs**
- OpenAI-style embedding APIs
- Azure OpenAI embedding deployments
- Amazon Titan Embeddings (via Bedrock)

### The Decision Framework

| Question | Why it matters |
| --- | --- |
| **Does my data leave the network?** | Regulated data may rule out API models entirely |
| **What languages?** | English-only models fail badly on multilingual corpora |
| **How long are my chunks?** | Check the max input length before you commit |
| **How many vectors will I store?** | Dimensions × 4 bytes × count = your RAM bill (§4) |
| **Latency budget?** | Embedding runs on *every* query, not just at indexing |
| **Prefix requirements?** | Get this wrong and quality silently degrades (§10) |

> 📊 Use the [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) to compare,
> but **evaluate on your own data**. MTEB measures general benchmarks; your legal or
> medical corpus is not a general benchmark. Build a small golden set of 30–50
> question→correct-chunk pairs and measure recall@k yourself. This takes an afternoon and
> is worth more than any leaderboard.

---

## 🎤 Interview Questions

**Q1. What is an embedding?**
> A dense numerical vector representing the semantic meaning of text, positioned so that
> similar meanings are close together in vector space.

**Q2. Why do we use embeddings instead of keyword search?**
> Keyword search matches exact strings and fails on synonyms and paraphrases. Embeddings
> match meaning, so "car" retrieves "automobile". In production you usually want both,
> because keyword search is still better for exact identifiers.

**Q3. What is cosine similarity?**
> A measure of the angle between two vectors, ranging from -1 to 1, used to estimate
> semantic similarity independently of vector magnitude.

**Q4. Cosine, dot product, or Euclidean — which should I use?**
> For normalised vectors, all three give identical rankings, so pick whichever your vector
> database computes fastest — usually dot product. The choice only matters for
> unnormalised vectors, where dot product biases toward larger magnitudes.

**Q5. How are embedding models trained?**
> Contrastive learning on triplets — anchor, positive, negative — pulling related pairs
> together and pushing unrelated pairs apart. Hard negatives matter most for quality.

**Q6. What are embeddings bad at?**
> Negation, exact identifiers, numeric comparison, and rare domain jargon. They capture
> the gist, not logic — which is why production systems add hybrid search, metadata
> filtering, and reranking.

**Q7. Can you change your embedding model after launch?**
> Only by re-embedding the entire corpus. Vectors from different models are incompatible,
> and mixing them returns confident nonsense rather than an error. So store the model name
> and version alongside every vector.

**Q8. Your RAG system misses documents that clearly contain the answer. How do you debug?**
> Check whether prefix conventions match between indexing and query time; check for
> silent truncation of chunks exceeding max input length; inspect the actual retrieved
> chunks rather than only the final answer; test whether the failures are exact-identifier
> or negation cases, which suggests hybrid search; and verify the same model version
> indexed and queried.

---

## ❌ Common Beginner Mistakes

- ❌ Thinking embeddings store the original text
- ❌ Assuming embeddings are random numbers
- ❌ Believing a larger dimension always means better performance
- ❌ Mixing up embeddings with tokens
- ❌ **Forgetting model-specific query prefixes** (§10)
- ❌ **Mixing vectors from different embedding models** (§10)
- ❌ Ignoring chunking strategy and blaming the embedding model
- ❌ Trusting embeddings on negation or exact identifiers
- ❌ Choosing a model from a leaderboard without testing on your own data

### 🔐 A Security Note

Embeddings do not *store* text — but they are not anonymous either. **Embedding inversion
attacks** can partially reconstruct source text from vectors. For GDPR and similar
regimes, treat your vector database as containing **personal data**: control access, and
make sure a deletion request removes the vectors too, not just the source document.

---

## 📝 Assignment

### Part 1 — Diagram

Draw the RAG pipeline and explain each step in your own words:

```text
User Question ──► Embedding Model ──► Vector Database ──► Top-k Chunks ──► LLM ──► Answer
```

### Part 2 — Questions

1. What is an embedding?
2. Why do embeddings enable semantic search?
3. Why is cosine similarity commonly used?
4. Why does RAG use embeddings instead of keyword matching?
5. What's the difference between a token and an embedding?

### Part 3 — Harder (Production Reality)

6. Your team wants to switch from `all-MiniLM-L6-v2` to a 1536-dimension model. What has
   to happen, and what does it cost?
7. Give two concrete queries where embeddings will fail and keyword search will succeed.
8. You have 5 million chunks. Compare RAM for a 384-dim vs a 1536-dim model. Show the maths.

### Part 4 — Practical

9. Run the numpy script from section 9 and confirm the three metrics agree on normalised
   vectors. Then remove the normalisation and observe the rankings diverge.
10. Install `sentence-transformers` and embed these three sentences:
    *"How can I lose weight?"*, *"Best diet plan for fat loss"*, *"How do I fix my car?"*
    Compute the similarity matrix. Do the results match your expectation?

Save your answers in [`assignments/lesson-04/`](../../assignments/lesson-04/).

---

## 📖 Resources

- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) — compare embedding models
- [Sentence-Transformers docs](https://www.sbert.net/) — the standard open-source library
- [BGE model card](https://huggingface.co/BAAI/bge-large-en-v1.5) — read the prefix requirements
- [E5 paper](https://arxiv.org/abs/2212.03533) — the `query:`/`passage:` convention explained
- [Text Embeddings Reveal (Almost) As Much As Text](https://arxiv.org/abs/2310.06816) — the embedding-inversion research behind the security note

---

## 📊 Progress Tracker

**Module 1: Modern AI Engineering**

- ✅ Lesson 1 – Role of a Production AI Engineer
- ✅ Lesson 2 – How LLMs Work
- ✅ Lesson 3 – Transformer Architecture
- ✅ Lesson 4 – Embeddings Deep Dive
- ⬜ Lesson 5 – Vector Databases & ANN Search

**Lessons remaining in Module 1:** 16
**Total bootcamp lessons remaining (approx.):** 116

---

## ➡️ What's Next – Lesson 5

**Vector Databases and Approximate Nearest Neighbour (ANN) Search**

- Why SQL databases aren't enough for semantic search
- How Qdrant, Weaviate, Pinecone, Milvus, and pgvector work
- HNSW indexing
- Top-k retrieval
- Metadata filtering
- **Hybrid search (BM25 + vectors)** — the fix for everything in section 12
- Performance optimisation for production RAG
- Live Python examples and enterprise interview questions

Nearly every production GenAI system stores embeddings in a vector database rather than a
relational one. This lesson explains why.

---

[⬅️ Lesson 3](lesson-03-transformers-architecture.md) · [🏠 Course home](../../README.md)
