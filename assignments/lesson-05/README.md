# Assignment – Lesson 5

## Task

Design a RAG system for a hospital with:

| Requirement | Value |
| --- | --- |
| Documents | **5 million** medical documents |
| Users | Doctors, searching in natural language |
| Metadata | Department, Country, Language, Publication Date |

Draw the architecture, then answer:

### Core Questions

1. Why would you choose a vector database?
2. Which retrieval method would you use?
3. Would you use hybrid search? Why?
4. How would metadata filtering improve performance?
5. Why is ANN preferable to exact search in this scenario?

### Harder

6. Compute the RAM needed for 5M vectors at 1024 dimensions. Does it fit on one machine?
7. A doctor filters to `Department = Cardiology`, which is 2% of documents. Explain what
   goes wrong with post-filtering.
8. Give one query where hybrid search returns the right document and pure vector search
   fails.

---

## ✅ My Solution

👉 **[solution.ipynb](solution.ipynb)** — a fully executed notebook containing:

- Ingestion and query architecture diagrams, with healthcare-specific additions
  (PHI handling, publication-date surfacing)
- Written answers to all eight questions
- **A working search engine over 1,000,000 vectors**, built and benchmarked:
  - Exact (brute-force) baseline vs an IVF ANN index
  - The measured recall/speed trade-off curve
  - The post-filter failure at 2% selectivity, demonstrated
  - Hybrid search: BM25 + vector + Reciprocal Rank Fusion
- Capacity planning maths for the real 40M-chunk corpus
- A production-readiness gap list

Every number in the notebook was produced by running the code. All cells are
pre-executed, so results are visible on GitHub without running anything.

---

## Three Results Worth Knowing

The notebook measures three things that are hard to believe until you see them, and all
three fail **silently** in production:

| Finding | Measured result |
| --- | --- |
| ANN vs exact search | ~**17× faster at 99% recall** — and the curve has a clear knee |
| Post-filtering at 2% selectivity | Asked for 10 results, averaged **0.22**; 40/50 queries returned **zero** |
| Vector search on an exact identifier | Returns **nothing**; BM25 finds it instantly |

---

## Self-Check

You have genuinely understood this lesson if you can explain, without notes:

- [ ] Why "5 million documents" is not 5 million vectors
- [ ] Why post-filtering silently returns fewer than k results, and why HTTP 200 makes
      that dangerous
- [ ] What `nprobe` (IVF) and `ef_search` (HNSW) have in common
- [ ] Why recall is a number you measure rather than a property you assume
- [ ] Why an exact identifier cannot be found by vector search, no matter how good the model
- [ ] Why RRF rewards consensus, and why a reranker is still needed after fusion
- [ ] When you would *not* use a vector database at all
