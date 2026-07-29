# Lesson 7 – Advanced Chunking Strategies

> **Module:** 2 – Retrieval-Augmented Generation (RAG)
> **Level:** Intermediate → Expert
> **Estimated time:** 75–90 minutes

**Goal:** Learn how companies like Microsoft, OpenAI, Anthropic, Google, Amazon, and
Thomson Reuters split documents for high-quality retrieval.

---

## 🚨 Why This Lesson Is Extremely Important

Many beginners believe:

> ❌ **Better LLM = Better RAG**

In production, **retrieval quality depends more on chunking than on the LLM**. A
frontier model with poor chunking will lose to a smaller model with excellent chunking —
because if the right text never reaches the model, no amount of intelligence recovers it.

Chunking is the cheapest, highest-leverage thing you can fix in a RAG system. It's also
the thing most teams never revisit after week one.

---

## 🗺️ Lesson Roadmap

1. What is chunking?
2. Why it matters
3. The nine strategies
4. **Measured comparison** — the strategies benchmarked on real text
5. Chunk overlap (and its hidden cost)
6. Choosing chunk size
7. ⚠️ The character-vs-token trap
8. Chunking by document type
9. Contextual Retrieval
10. How to actually evaluate chunking
11. Interview questions

---

## 1. What Is Chunking?

You have a 300-page employee handbook. Should you send all 300 pages to the LLM?

❌ No — it's too expensive, too slow, exceeds the context window, and (per Lesson 3's
O(n²) attention) most of it is irrelevant noise that actively degrades the answer.

So we split it:

```text
PDF  ──►  Chunk 1  ·  Chunk 2  ·  Chunk 3  ·  Chunk 4
```

Each chunk is embedded and becomes independently searchable.

### In Simple Terms

Chunking means dividing a large document into smaller logical pieces, so the AI retrieves
only the part it actually needs.

### Example

```text
Page 1 – Company History     Page 3 – Leave Policy
Page 2 – Office Rules        Page 4 – Remote Work
```

User asks: *"Can employees work remotely?"*

The AI needs **page 4 only**. Everything else is cost and distraction.

### Why It Matters

| Good chunking | Bad chunking |
| --- | --- |
| ✅ Accuracy | ❌ Missing answers |
| ✅ Speed | ❌ Wrong answers |
| ✅ Lower cost | ❌ Hallucinations |
| ✅ Better recall & precision | ❌ Higher token bills |

> 🔑 **The core tension, in one sentence:** small chunks retrieve *precisely* but lose
> context; large chunks carry context but retrieve *vaguely*. Every strategy below is an
> attempt to get both.

---

## 2. The Nine Strategies

### Method 1 — Fixed-Size Chunking

Split every N characters or tokens, regardless of content.

```text
500 words  │  500 words  │  500 words
```

✅ Trivial to implement, fast, predictable
❌ Splits sentences — and sometimes **words** — in half

```text
Chunk 1: "...Employees can request leave"
Chunk 2: "only after manager approval."
```

Neither chunk states the actual rule. Section 3 shows this failing with real measurements.

### Method 2 — Sentence Chunking

Split on sentence boundaries. Much safer, but a single idea often spans several sentences.

### Method 3 — Paragraph Chunking

One paragraph → one chunk. Common and effective for HR policies, legal documents, and
articles, where paragraphs are already semantic units.

### Method 4 — Recursive Chunking ⭐ *the production default*

Try separators in order of decreasing semantic strength, until chunks fit:

```text
Paragraph (\n\n)  →  Line (\n)  →  Sentence (". ")  →  Word (" ")  →  Character
```

If a paragraph fits, keep it whole. If not, break it into sentences. Still too big? Words.
**It only ever degrades as far as it must**, preserving the strongest boundary available.

This is the right default for most projects, and the one to name in an interview.

### Method 5 — Semantic Chunking

Embed each sentence, then cut where consecutive sentences become dissimilar — i.e. where
the topic changes.

```text
History │ Company Values │ Leave Policy │ Remote Work
```

⚠️ **An honest assessment**, because most tutorials oversell this: semantic chunking
requires embedding **every sentence** in your corpus before you can chunk it. On millions
of documents that's a large, recurring bill, and the measured improvement over recursive
chunking is often modest. Recursive chunking plus good headings and metadata frequently
matches it for a fraction of the cost.

**Use it when** document structure is genuinely absent (transcripts, OCR output, scraped
text with no headings). **Don't reach for it first** on well-structured documents.

### Method 6 — Parent–Child Chunking ⭐ *small-to-big*

Also called *auto-merging retrieval*. Index small chunks, but return their larger parent:

```text
Parent: "Annual HR Policy" (2000 tokens)
   ├── Child: Leave Policy      (200 tokens)  ← these are indexed & searched
   ├── Child: Remote Work       (200 tokens)
   └── Child: Insurance         (200 tokens)

Search the CHILD  ──►  return the PARENT to the LLM
```

**This directly resolves the core tension.** Small chunks give precise retrieval; large
parents give the LLM full context. It's one of the highest-value upgrades to a basic RAG
system, and worth mentioning in any interview.

### Method 7 — Sliding Window Chunking

Overlapping windows:

```text
Chunk 1: ABCDE
Chunk 2:   CDEFG
Chunk 3:     EFGHI
```

Ideas that continue across a boundary remain findable.

### Method 8 — Agentic Chunking

An LLM reads the document and decides the boundaries itself — where topics change, which
sections belong together, what metadata to attach.

⚠️ Powerful, but you are running an LLM over **every document you own**. Reserve it for
high-value corpora where retrieval quality justifies the cost, not for a general pipeline.

### Method 9 — Contextual / Structure-Aware Chunking

Don't store a clause naked. Store it with its lineage:

```text
❌ "The rate shall not exceed 4.5%."

✅ "Employment Contract 2026 > Section 5: Tax Rules > Clause 5.2:
    The rate shall not exceed 4.5%."
```

The first chunk is nearly unretrievable — *what* rate? The second embeds far more
accurately, and the citation is meaningful to the user.

> 💡 **The cheapest quality win in all of RAG:** prepend the document title and section
> heading to every chunk before embedding. It costs almost nothing and often improves
> retrieval more than switching embedding models.

---

## 3. 📊 Measured Comparison

Claims about chunking are easy. Here is an actual measurement on a short HR handbook,
with five questions whose answers we know.

A question is **answerable** only if its complete answer span sits inside a single chunk.

```python
def recursive_chunk(text, size=250, separators=None):
    """Try separators in order of decreasing semantic strength."""
    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]
    if len(text) <= size:
        return [text] if text.strip() else []

    sep, rest = separators[0], separators[1:]
    if sep == "":
        return [text[i:i + size] for i in range(0, len(text), size)]

    chunks, current = [], ""
    for part in text.split(sep):
        candidate = (current + sep + part) if current else part
        if len(candidate) <= size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(part) > size:                      # still too big → go finer
                chunks.extend(recursive_chunk(part, size, rest))
                current = ""
            else:
                current = part
    if current:
        chunks.append(current)
    return [c for c in chunks if c.strip()]
```

**Real measured output:**

```text
strategy                 chunks   start mid-sent     answerable
----------------------------------------------------------------
Fixed (no overlap)            4          3 /4             4 /5
Fixed (50 overlap)            4          2 /4             5 /5
Sentence                      4          0 /4             5 /5
Recursive                     5          0 /5             5 /5

Which questions each strategy can answer:
  Fixed (no overlap)     missed: ['How much notice for parental leave?']
  Fixed (50 overlap)     missed: none
  Sentence               missed: none
  Recursive              missed: none
```

### Look at What Fixed Chunking Actually Did

```text
--- chunk 2 ---  '...must be taken within six months of the birth. Applications require 8 wee'
--- chunk 3 ---  'ks notice.'
```

It split **mid-word**: `8 wee` | `ks notice`. The document plainly states the notice
period, and no chunk contains that fact. The system will answer *"I could not find
information on that"* — and it will look like a knowledge gap rather than a chunking bug.

### Three Conclusions

1. **Fixed chunking loses real answers.** 3 of 4 chunks began mid-sentence, and one
   question of five became unanswerable from a document that clearly answers it.
2. **Overlap is a patch, not a fix.** It rescued the missing answer, but 2 of 4 chunks
   still start mid-sentence — meaning embeddings are still computed over fragments.
3. **Recursive produced one more chunk and lost nothing.** Slightly more storage, zero
   broken boundaries. That trade is why it's the default.

> ⚠️ **Scope, stated honestly:** this is a short document with clean structure, and the
> effect is deliberately visible. It demonstrates *the failure mode*, not a benchmark
> score you should expect on your corpus. Run this measurement on **your** data — the
> method is the transferable part.

---

## 4. Chunk Overlap

Without overlap, a fact spanning a boundary is lost by both chunks:

```text
Chunk 1: "Employees are eligible for"
Chunk 2: "annual leave after six months."
```

With overlap, both remain findable.

### How Much?

| Chunk size | Typical overlap |
| --- | --- |
| 300 tokens | ~50 tokens |
| 500 tokens | ~75 tokens |
| 1000 tokens | 100–200 tokens |

Roughly **10–20%**. There is no universal rule — evaluate on your own data.

### ⚠️ The Hidden Costs Nobody Mentions

Overlap is not free, and three consequences bite in production:

1. **Storage and cost scale with it.** 20% overlap means ~20% more chunks: more vectors,
   more RAM, more embedding API calls. Revisit the storage maths from Lesson 4.
2. **Duplicate retrieval results.** The same sentence lives in two chunks, so both can
   surface in the top-k. You just spent two of your five slots on identical text.
   **Deduplicate after retrieval** — this is a real bug in many systems.
3. **It's a patch for bad boundaries.** With recursive or structure-aware chunking your
   boundaries already fall in sensible places, so you need far less overlap. Heavy
   reliance on overlap usually signals the wrong splitter.

---

## 5. Choosing Chunk Size

### The Trade-off

| Small chunks (~100 tokens) | Large chunks (~1500 tokens) |
| --- | --- |
| ✅ Precise retrieval | ✅ Complete context |
| ❌ Missing context | ❌ Higher cost |
| ❌ Answer may need several chunks | ❌ Dilutes the embedding (Lesson 4) |

### Typical Starting Points

| Document type | Suggested size |
| --- | --- |
| FAQs | 200–400 tokens (often one Q&A pair) |
| HR policies | 400–700 tokens |
| Legal documents | 700–1200 tokens |
| Research papers | 800–1500 tokens |
| Source code | By function/class, not by token count |

These are **starting points**. Measure and adjust.

---

## 6. ⚠️ The Character-vs-Token Trap

This one silently misconfigures a very large number of RAG systems.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,        # ⚠️ 500 CHARACTERS — not tokens!
    chunk_overlap=75,
)
```

It's called `RecursiveCharacter…` for a reason: by default it measures length with
`len()`, which counts **characters**. From Lesson 2, roughly 4 characters ≈ 1 token, so:

```text
chunk_size = 500 characters  ≈  125 tokens
```

You intended 500-token chunks and built 125-token ones — **a quarter of the size**. Your
chunks are too small, context is missing, and retrieval quietly underperforms while
everything appears configured correctly.

### The Fix

```python
splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",
    chunk_size=500,        # ✅ now genuinely 500 TOKENS
    chunk_overlap=75,
)
```

> 🔑 **Always check what unit your splitter measures in**, and log the actual token count
> of your chunks after building the index. One `print()` of the mean and max token count
> catches this immediately.

---

## 7. Chunking by Document Type

**One strategy for all document types is the most common structural mistake.**

### ⚖️ Legal Documents
One clause or section per chunk. **Never split a clause** — half a clause can invert its
legal meaning. Keep the section heading and clause number attached.

### 🏥 Medical Records
Keep diagnosis, treatment, and medication **together**. Split apart, a chunk saying
*"administer 50mg twice daily"* has lost the condition it treats — actively dangerous.

### 💻 Source Code
Chunk by **function, class, or module** — never by line count. A function split in half is
useless to both retrieval and the model. Keep signatures and docstrings with their bodies,
and attach the file path as metadata.

### 📄 Research Papers
Split by section: Abstract, Introduction, Methods, Results, Discussion. Each answers a
different kind of question.

### 📊 Tables
Never split a table across chunks — the rows lose their headers and become meaningless.
Keep small tables whole; for large ones, repeat the header row in each piece, or convert
to a text summary and store the table as metadata.

---

## 8. Contextual Retrieval

A notable recent refinement of Method 9, published by Anthropic. Instead of only
prepending headings, use an LLM to generate a short situating description for each chunk
at indexing time:

```text
Original chunk:
  "The rate shall not exceed 4.5%."

Contextualised chunk (what actually gets embedded):
  "This chunk is from the 2026 Employment Contract, Section 5 on Tax Rules,
   discussing the maximum withholding rate for contractors.
   The rate shall not exceed 4.5%."
```

The reported result is a substantial reduction in retrieval failures. The trade is the
same as agentic chunking — **one LLM call per chunk at index time** — but it's a one-off
ingestion cost rather than a per-query cost, and prompt caching makes it far cheaper than
it first appears.

**When it's worth it:** high-value corpora where a missed retrieval is expensive — legal,
medical, financial, compliance.

---

## 9. How to Actually Evaluate Chunking

Everything above is a hypothesis until you measure it. **You cannot pick a chunking
strategy by reading about chunking strategies.**

### Build a Golden Set

30–50 realistic questions with their correct source chunk. An afternoon of work, and it
becomes the foundation for every future decision.

```python
GOLDEN = [
    {"question": "How much maternity leave?", "must_contain": "26 weeks"},
    {"question": "Can I work remotely?",      "must_contain": "three days per week"},
    # ... 30-50 of these
]
```

### Then Sweep

```python
for size in [300, 500, 800, 1200]:
    for overlap in [0, 50, 100]:
        chunks = build_index(documents, size, overlap)
        score = recall_at_k(GOLDEN, chunks, k=5)
        print(size, overlap, score)
```

Two hours of compute answers a question that people argue about for weeks.

### Measure These

| Metric | Question it answers |
| --- | --- |
| **Recall@k** | Is the right chunk in the top-k at all? |
| **Answer completeness** | Does one chunk contain the *full* answer? |
| **Chunk count** | What does this cost to store and embed? |
| **Mean/max tokens** | Are chunks the size I actually intended? (§6) |

> 🔗 Recall Lesson 5 §10: **index recall** (is ANN finding what brute force would?) is a
> different thing from **retrieval quality** (is the right chunk in the top-k?). Chunking
> affects the second. If index recall is 0.99 and answers are still wrong, look here.

---

## 🏆 Production Best Practices

- ✅ Default to **recursive chunking**; upgrade to parent–child when you need both precision and context
- ✅ **Prepend document title and section headings** to every chunk — cheapest win available
- ✅ Store rich metadata: document ID, page, title, author, date, department, section
- ✅ Keep tables, code blocks, and legal clauses **intact**
- ✅ Use a **different strategy per document type**
- ✅ **Log actual token counts** after indexing (§6)
- ✅ **Deduplicate retrieved chunks** when using overlap (§4)
- ✅ Build a golden set and measure before and after every change

## ❌ Common Mistakes

- ❌ Tiny chunks that lose context
- ❌ Enormous chunks that dilute the embedding
- ❌ No overlap and no structure awareness
- ❌ Splitting tables, code, or clauses randomly
- ❌ Ignoring document structure entirely
- ❌ One strategy for every document type
- ❌ **Confusing characters with tokens** (§6)
- ❌ **Never deduplicating overlapping results**
- ❌ Changing chunking without measuring the effect

---

## 🎤 Interview Questions

**Q1. Why is chunking important?**
> It determines what can be retrieved at all. If the right text never makes it into a
> retrievable chunk, no model can recover it — so chunking often affects RAG quality more
> than model choice.

**Q2. Why is recursive chunking popular?**
> It preserves the strongest available semantic boundary, only degrading to finer
> separators when a chunk still exceeds the size limit. Good structure preservation for
> almost no cost.

**Q3. What is semantic chunking, and when would you avoid it?**
> Splitting where the topic changes, detected by embedding sentences and finding
> dissimilarity. I'd avoid it on well-structured documents at scale — it means embedding
> every sentence, and recursive chunking with good headings often matches it far cheaper.

**Q4. Why use overlap, and what does it cost?**
> It preserves facts spanning boundaries. It costs proportionally more storage and
> embedding calls, and it causes duplicate retrieval results that must be deduplicated.

**Q5. When would you use parent–child chunking?**
> When precise retrieval and rich context are both needed. Index small chunks for accurate
> matching, then return the larger parent to the LLM.

**Q6. Same strategy for code and legal documents?**
> No. Code is chunked by function or class; legal text by clause or section, never
> splitting a clause, since half a clause can reverse its meaning.

**Q7. How do you know your chunking is good?**
> Build a golden set of question→correct-chunk pairs and measure recall@k plus answer
> completeness, sweeping chunk size and overlap. Without that, it's guesswork.

**Q8. Your chunks are 500 "size" but retrieval lacks context. What do you check first?**
> Whether the splitter measures characters or tokens. `RecursiveCharacterTextSplitter`
> defaults to characters, so `chunk_size=500` is about 125 tokens — a quarter of what was
> intended.

---

## 💼 Real Enterprise Interview Scenario

> **Interviewer:** You have 25 million legal documents. Users complain the chatbot
> retrieves incomplete clauses. How would you improve retrieval?

A strong answer:

1. **Measure first** — build a golden set of queries with known correct clauses and
   establish a baseline. Without it, every change is guesswork.
2. **Inspect actual chunk boundaries** — are clauses being split? Are chunks the size we
   think they are, or did characters-vs-tokens halve them?
3. **Switch to structure-aware chunking** — one clause per chunk, never split; prepend
   document title, section, and clause number before embedding.
4. **Add parent–child retrieval** — match on the clause, return the full section so the
   model sees surrounding definitions.
5. **Add a reranker** — retrieve 50, rerank to 5.
6. **Add hybrid search** — legal queries cite exact references like "Section 15(b)", which
   embeddings blur (Lesson 5).
7. **Re-measure** and report the delta on the golden set.

> The structure of that answer matters as much as the content: **measure → diagnose →
> fix → re-measure.** Candidates who jump straight to "use semantic chunking" sound like
> they've read a blog post. Candidates who ask what the baseline is sound like engineers.

---

## 📝 Assignment

Design a chunking strategy for each document type:

1. 📚 University textbooks
2. ⚖️ Legal contracts
3. 💻 Python source code
4. 🏥 Medical records
5. ❓ FAQ documents

For each, specify:

- Which chunking strategy, and why it fits this document type
- Approximate chunk size
- Overlap
- What metadata you would attach
- One concrete failure that would occur with the *wrong* strategy

**Harder:**

6. You set `chunk_size=500` and retrieval lacks context. Give two distinct causes.
7. Overlap is 20% and users report duplicated text in answers. Diagnose and fix.
8. Design an experiment to decide between 500-token and 1000-token chunks. What do you
   measure, and what result would change your mind?

Save your answers in [`assignments/lesson-07/`](../../assignments/lesson-07/).

---

## 📖 Resources

- [LangChain Text Splitters](https://python.langchain.com/docs/concepts/text_splitters/)
- [LlamaIndex Node Parsers](https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/) — parent–child / auto-merging
- [Anthropic — Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
- [Chunking Strategies for LLM Applications (Pinecone)](https://www.pinecone.io/learn/chunking-strategies/)

---

## 📊 Progress Tracker

**Module 2 – Retrieval-Augmented Generation (RAG)**

- ⬜ Lesson 6 – RAG from Beginner to Production *(not yet in this repo — see README)*
- ✅ Lesson 7 – Advanced Chunking Strategies
- ⬜ Lesson 8 – Embedding Models Masterclass

**Lessons remaining in Module 2:** 18
**Overall bootcamp remaining (approx.):** 113 lessons

---

## ➡️ What's Next – Lesson 8

**Embedding Models Masterclass**

How embedding models are trained · SentenceTransformers architecture · BGE, E5, Jina,
Nomic, Voyage · dense vs sparse embeddings · multilingual embeddings · choosing and
benchmarking a model · fine-tuning · production architectures · hands-on coding.

Choosing the right embedding model can affect a RAG system as much as choosing the right
LLM.

---

[⬅️ Lesson 5](../module-01-modern-ai-engineering/lesson-05-vector-databases-and-ann-search.md) · [🏠 Course home](../../README.md)
