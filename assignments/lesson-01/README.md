# Assignment – Lesson 1

## Task

Draw the architecture of an AI chatbot that answers questions from **uploaded PDF
documents**.

Your diagram must include all nine components:

1. User
2. Frontend
3. FastAPI backend
4. Document storage
5. Embedding model
6. Vector database
7. LLM
8. Safety layer
9. Response back to the user

## ✅ My Solution

👉 **[solution.ipynb](solution.ipynb)** — a Jupyter notebook containing:

- Both architecture flows (ingestion and query) as text diagrams
- A complete combined system diagram
- A component-by-component checklist of all nine required parts
- **A runnable simulation** of the entire pipeline with no external dependencies
- Four safety-layer tests: prompt injection, cross-tenant access, unanswerable
  questions, and hallucination detection
- What is still missing before this would be production-ready

All cells are pre-executed, so the outputs are visible directly on GitHub without
running anything.

## Hints

- There are really **two flows**, not one. Think about what happens when a PDF is
  *uploaded* versus what happens when a question is *asked*.
- The upload flow happens once per document. The query flow happens on every question.
- Ask yourself where each of these belongs: where is the PDF text extracted? Where are
  embeddings created? What is stored, and where?

## Self-Check

Once you're done, check your diagram against these questions:

- [ ] Have I shown the ingestion (upload) flow separately from the query flow?
- [ ] Is it clear *where* the PDF text gets extracted and split into chunks?
- [ ] Does the LLM receive the retrieved context, rather than the whole document store?
- [ ] Does the safety layer sit **between** the LLM and the user?
- [ ] Could another engineer build the system from my diagram alone?
