# 🎓 Production GenAI Engineer Bootcamp (2026)

An industry-level bootcamp that takes you from **Intermediate → Expert** and prepares you
for AI Engineer roles at companies like **Andela, Microsoft, Thomson Reuters, Amazon,
Google, Meta**, and AI-first startups.

This is **not** a typical tutorial series. Every lesson is written the way an enterprise
AI team would actually onboard a new engineer: real architecture, real trade-offs, real
production concerns.

---

## 📌 Course Details

| | |
| --- | --- |
| **Level** | Intermediate → Expert |
| **Goal** | Become a Production AI Engineer |
| **Duration** | ~120 lessons |
| **Language** | English |
| **Author** | [Naeem Naseer](https://github.com/Naeem2400) |

---

## 📚 What Every Lesson Includes

- ✅ Clear English explanation
- ✅ Simple, plain-language summary
- ✅ Real-world examples from production systems
- ✅ Text-based architecture diagrams
- ✅ Interview questions with strong answers
- ✅ Coding examples
- ✅ Best practices
- ✅ Common mistakes to avoid
- ✅ Mini assignment
- ✅ Further resources

---

## 🗺️ Curriculum Roadmap

| Module | Title | Covers | Status |
| --- | --- | --- | --- |
| 1 | **Modern AI Engineering** | The role, LLM internals, tokens, embeddings, transformers, modern Python | 🟡 In Progress |
| 2 | **Retrieval-Augmented Generation (RAG)** | Chunking, embedding models, retrieval, reranking, evaluation | 🟡 In Progress |
| 3 | Embeddings & Vector Databases | Chunking, indexing, Qdrant/Weaviate/pgvector, hybrid search | ⬜ Planned |
| 4 | Retrieval-Augmented Generation (RAG) | Retrieval pipelines, reranking, citations, advanced RAG | ⬜ Planned |
| 5 | Building AI APIs with FastAPI | Pydantic contracts, streaming, async, background jobs | ⬜ Planned |
| 6 | AI Agents & Tool Use | Tool calling, planning loops, multi-agent systems, MCP | ⬜ Planned |
| 7 | Evaluation, Testing & Observability | Golden sets, LLM-as-judge, tracing, regression testing | ⬜ Planned |
| 8 | Guardrails, Safety & Security | Injection defence, PII, grounding checks, multi-tenancy | ⬜ Planned |
| 9 | Deployment, Docker & Cloud | Docker, AWS ECS, Azure AKS, Kubernetes, CI/CD | ⬜ Planned |
| 10 | Cost, Latency & Scaling | Caching, batching, model routing, token budgets | ⬜ Planned |
| 11 | Capstone Projects & Interview Preparation | End-to-end builds, system design, mock interviews | ⬜ Planned |

> Module 1 runs to roughly 20 lessons. The module list is a working plan and will be
> refined as the bootcamp progresses.

---

## ✅ Lesson Progress

### Module 1 – Modern AI Engineering

| # | Lesson | Link | Assignment | Status |
| --- | --- | --- | --- | --- |
| 1 | What Does a Production AI Engineer Actually Do? | [Read](lessons/module-01-modern-ai-engineering/lesson-01-what-does-a-production-ai-engineer-do.md) | [Solution](assignments/lesson-01/solution.ipynb) | ✅ Done |
| 2 | How LLMs Actually Work | [Read](lessons/module-01-modern-ai-engineering/lesson-02-how-llms-actually-work.md) | [Brief](assignments/lesson-02/) | ✅ Done |
| 3 | Transformers: The Technology That Changed AI Forever | [Read](lessons/module-01-modern-ai-engineering/lesson-03-transformers-architecture.md) | [Brief](assignments/lesson-03/) | ✅ Done |
| 4 | Embeddings Deep Dive (The Foundation of RAG) | [Read](lessons/module-01-modern-ai-engineering/lesson-04-embeddings-deep-dive.md) | [Brief](assignments/lesson-04/) | ✅ Done |
| 5 | Vector Databases & ANN Search | [Read](lessons/module-01-modern-ai-engineering/lesson-05-vector-databases-and-ann-search.md) | [Solution](assignments/lesson-05/solution.ipynb) | ✅ Done |

### Module 2 – Retrieval-Augmented Generation (RAG)

| # | Lesson | Link | Assignment | Status |
| --- | --- | --- | --- | --- |
| 6 | RAG from Beginner to Production | — | — | ⚠️ **Missing** |
| 7 | Advanced Chunking Strategies | [Read](lessons/module-02-retrieval-augmented-generation/lesson-07-advanced-chunking-strategies.md) | [Brief](assignments/lesson-07/) | ✅ Done |
| 8 | Embedding Models Masterclass | *Coming next* | — | ⬜ |

> ⚠️ **Lesson 6 gap:** the course jumped from Lesson 5 to Lesson 7, so Lesson 6 — *RAG
> from Beginner to Production* — is not yet written. Lesson 7 assumes it. It should cover
> the end-to-end RAG pipeline, ingestion, retrieval, reranking, prompt construction,
> evaluation, and common production failures.

---

## 📂 Repository Structure

```text
AI-Engineer-full-course/
├── README.md                 # You are here
├── lessons/                  # All lesson notes, organised by module
│   ├── module-01-modern-ai-engineering/
│   │   ├── lesson-01-what-does-a-production-ai-engineer-do.md
│   │   ├── lesson-02-how-llms-actually-work.md
│   │   ├── lesson-03-transformers-architecture.md
│   │   ├── lesson-04-embeddings-deep-dive.md
│   │   └── lesson-05-vector-databases-and-ann-search.md
│   └── module-02-retrieval-augmented-generation/
│       └── lesson-07-advanced-chunking-strategies.md
├── assignments/              # Assignment briefs + my solutions
│   ├── lesson-01/            # brief + solution.ipynb
│   ├── lesson-02/            # brief
│   ├── lesson-03/            # brief
│   ├── lesson-04/            # brief
│   ├── lesson-05/            # brief + solution.ipynb
│   └── lesson-07/            # brief
└── projects/                 # Larger end-to-end projects (added later)
```

---

## 🛠️ Core Skills Covered

| Skill | Why It Matters |
| --- | --- |
| Python | Main programming language |
| FastAPI | Build AI APIs |
| SQL | Store structured data |
| Vector Databases | Semantic search |
| LLMs | Generate responses |
| RAG | Use company knowledge |
| Docker | Package applications |
| Git | Team collaboration |
| AWS / Azure | Deploy and scale |
| Testing | Ensure reliability |
| Monitoring | Keep systems healthy |

---

## 🚀 How to Use This Repository

1. Read the lessons in order — each one builds on the previous.
2. Complete the **mini assignment** at the end of every lesson before moving on.
3. Commit your assignment solutions into `assignments/lesson-XX/`.
4. Revisit the **Interview Questions** sections before any real interview.

---

## 📄 License

Released under the [MIT License](LICENSE). Free to learn from, fork, and share.
