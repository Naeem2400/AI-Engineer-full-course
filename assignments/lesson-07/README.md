# Assignment – Lesson 7

## Task

Design a chunking strategy for each of these document types:

1. 📚 University textbooks
2. ⚖️ Legal contracts
3. 💻 Python source code
4. 🏥 Medical records
5. ❓ FAQ documents

For each one, specify:

| Field | What to give |
| --- | --- |
| **Strategy** | Which chunking method, and why it fits |
| **Chunk size** | Approximate, in **tokens** (state the unit!) |
| **Overlap** | How much, and why |
| **Metadata** | What you'd attach to each chunk |
| **Failure mode** | One concrete thing that breaks with the *wrong* strategy |

---

## Harder

6. You set `chunk_size=500` and retrieval still lacks context. Give **two distinct**
   causes and how you'd tell them apart.
7. Your overlap is 20% and users report the same text appearing twice in answers.
   Diagnose the cause and give the fix.
8. Design an experiment to decide between 500-token and 1000-token chunks. What exactly
   do you measure, and what result would change your mind?

---

## How to Submit

Add your answers to this folder as either:

- `solution.md` — a table per document type, or
- `solution.ipynb` — a notebook, if you want to implement and compare the splitters

---

## Self-Check

You have genuinely understood this lesson if you can explain, without notes:

- [ ] The core tension: small chunks retrieve precisely but lose context; large chunks
      carry context but retrieve vaguely
- [ ] Why recursive chunking is the sensible default
- [ ] How parent–child chunking resolves the core tension
- [ ] Why semantic chunking is often **not** worth its cost
- [ ] The three hidden costs of overlap — storage, duplicate results, and masking bad boundaries
- [ ] Why `chunk_size=500` in `RecursiveCharacterTextSplitter` is about 125 tokens
- [ ] Why you cannot choose a chunking strategy without a golden set

> 💡 **The single highest-value habit from this lesson:** prepend the document title and
> section heading to every chunk before embedding. It costs almost nothing and often
> improves retrieval more than changing embedding models.
