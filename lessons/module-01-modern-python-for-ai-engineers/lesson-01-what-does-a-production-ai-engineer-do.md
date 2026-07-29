# Lesson 1 – What Does a Production AI Engineer Actually Do?

> **Module:** 1 – Modern Python for AI Engineers
> **Level:** Intermediate → Expert
> **Estimated time:** 45–60 minutes

This is one of the most important lessons in the entire bootcamp, because most people
completely misunderstand what this role actually is. If you get this wrong, you will
spend months learning the wrong things.

---

## 🏢 Imagine You're Working at Thomson Reuters

The company holds an enormous amount of information:

- Millions of legal documents
- Millions of tax documents
- Medical records
- Financial reports
- Compliance documents

One day, a lawyer types a question into the internal AI assistant:

> *"What are the latest regulations for corporate tax in the UK?"*

The system is expected to return an accurate, source-backed answer **in seconds**.

The obvious question is: **how?**

---

## ❌ What Beginners Think Happens

Most beginners picture something like this:

```text
User
 ↓
ChatGPT
 ↓
Answer
```

This is **not** how enterprise AI systems work. Not even close.

Sending a raw user question straight to a model gives you an answer that is:

- Not grounded in the company's own documents
- Impossible to audit or cite
- Potentially confidently wrong (a hallucination)
- A compliance and data-leakage risk

---

## ✅ What Actually Happens in Production

A real production system looks much more like this:

```text
                User
                  │
                  ▼
            FastAPI Backend
                  │
                  ▼
        Authentication & Security
                  │
                  ▼
         Prompt Builder / Context Engine
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
Vector Database        Business Database
(Qdrant/Weaviate)      (PostgreSQL)
        │                   │
        └─────────┬─────────┘
                  ▼
              LLM (GPT/Claude/Llama)
                  │
                  ▼
        Safety & Validation Layer
                  │
                  ▼
              Final Response
```

Notice that the LLM is only **one box** out of eight. This is the single biggest mental
shift you need to make: **the model is a component, not the product.**

---

## 🧠 Simple Explanation

Let's walk through the same example in plain language.

A user asks:

> *"Tell me about UK tax law."*

The AI does **not** answer immediately. Instead:

1. **The user is verified.** Are they logged in? Are they allowed to see tax documents?
   A junior clerk and a senior partner may not have the same access.
2. **Relevant documents are searched.** The system finds the handful of documents that
   actually relate to UK corporate tax, out of millions.
3. **A prompt is built.** The question and the retrieved documents are combined into a
   carefully structured instruction.
4. **Only the necessary context is sent to the LLM.** Not the whole database — just what
   matters. This controls both cost and accuracy.
5. **The output passes through safety checks.** Did it leak private data? Did it invent a
   regulation that doesn't exist? Is it citing real sources?
6. **The final answer is returned to the user**, usually with citations.

This whole process is called a **production AI pipeline**. Your job as an AI Engineer is
to design, build, and maintain that pipeline.

---

## 🔨 What Does an AI Engineer Build?

An AI Engineer does not just write prompts. They build complete systems such as:

- AI chatbots
- Document search engines
- Legal assistants
- Medical assistants
- AI coding assistants
- Customer support bots
- Voice assistants
- AI agents
- Enterprise search
- Recommendation systems

---

## 💼 Responsibilities in a Real Job

Imagine your manager walks over and says:

> *"Build an AI assistant for our lawyers."*

Here is what you would actually do.

### Step 1 – Understand the Business

Before writing a single line of code, ask:

- Who will use it?
- What problem are we solving?
- How accurate must it be?
- What regulations apply (GDPR, HIPAA, internal compliance)?
- What does "good enough to ship" look like?

> 💡 **This step separates junior from senior engineers.** Juniors start coding. Seniors
> start asking questions.

### Step 2 – Collect Data

Your knowledge sources might include:

- PDFs
- Word files
- Web pages
- Databases
- Emails
- Internal policies

### Step 3 – Clean the Data

Real-world data is messy. You need to remove:

- Duplicates
- Empty pages
- Broken or garbled text
- Headers, footers, and other noise

> Garbage in, garbage out. Poor data cleaning is the number one cause of poor RAG quality.

### Step 4 – Generate Embeddings

Convert text into vectors — lists of numbers that capture *meaning*, so that
"corporation tax" and "company tax rate" are recognised as related even though the words
differ.

We will cover exactly how this works in a later module.

### Step 5 – Store in a Vector Database

Common choices:

- Qdrant
- Weaviate
- Pinecone
- pgvector (PostgreSQL extension)

### Step 6 – Build APIs

Using:

- Python
- FastAPI
- Pydantic (for validation and typed contracts)

### Step 7 – Connect the LLM

Options include:

- GPT
- Claude
- Llama
- Qwen
- Mistral

### Step 8 – Add Guardrails

Protect against:

- Hallucinations
- Prompt injection
- Harmful responses
- Data leakage

### Step 9 – Test

Measure:

- **Accuracy** — is the answer correct and grounded?
- **Latency** — how long does the user wait?
- **Cost** — how much does each query cost in tokens?
- **Safety** — does it fail gracefully on adversarial input?

### Step 10 – Deploy

Using:

- Docker
- AWS ECS
- Azure AKS
- Kubernetes

---

## 🧰 Core Skills of a Production AI Engineer

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

## 💻 Coding Example – The Shape of a Production Endpoint

You don't need to understand every line yet. The goal is to *see the structure* — notice
how little of it is about the LLM itself.

```python
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Legal AI Assistant")


class Question(BaseModel):
    """Validated input contract. Bad requests are rejected before any cost is incurred."""
    query: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)


class Answer(BaseModel):
    """Validated output contract, including the sources used."""
    answer: str
    sources: list[str]


@app.post("/ask", response_model=Answer)
def ask(question: Question, user=Depends(get_current_user)):
    # 1. Authorisation — can this user see these documents?
    if not user.can_read("legal"):
        raise HTTPException(status_code=403, detail="Not authorised")

    # 2. Retrieval — find only the relevant context
    documents = vector_db.search(question.query, top_k=question.top_k)

    # 3. Prompt building — structure the context for the model
    prompt = build_prompt(question.query, documents)

    # 4. Generation — the LLM call is a single step, not the whole system
    raw_answer = llm.generate(prompt)

    # 5. Validation — block hallucinations and leaks before the user sees them
    safe_answer = guardrails.validate(raw_answer, documents)

    return Answer(answer=safe_answer, sources=[d.source for d in documents])
```

**Count the lines.** Exactly one of them calls the LLM. The other 90% of the work is
engineering: validation, authorisation, retrieval, and safety. That ratio is a fair
picture of the real job.

---

## 🎤 Interview Question

**Q: What is the difference between an AI Researcher and an AI Engineer?**

**A strong answer:**

> An AI Researcher focuses on creating or improving models and algorithms, usually
> through experimentation and publication. Their output is new knowledge — a better
> architecture, a novel training method, a paper.
>
> An AI Engineer focuses on building reliable, scalable, production-ready applications
> using AI models. That includes APIs, data pipelines, retrieval systems, deployment,
> monitoring, evaluation, and ongoing maintenance. Their output is a working system that
> real users depend on.
>
> In short: the researcher improves the model; the engineer makes the model useful,
> safe, fast, and affordable in production.

**Follow-up questions interviewers often ask next:**

- "Where would you add caching in that pipeline, and why?"
- "How would you know if your system's answers got worse after a model upgrade?"
- "How do you stop a user from extracting another customer's documents?"

You aren't expected to answer these yet — but notice they are all *engineering*
questions, not model questions.

---

## ⚠️ Common Mistakes Beginners Make

- ❌ Learning only prompt engineering
- ❌ Ignoring Python fundamentals
- ❌ Never deploying a project
- ❌ Not learning how to build APIs
- ❌ Ignoring databases
- ❌ Believing that "using ChatGPT" is the same as "building AI systems"
- ❌ Skipping evaluation — if you can't measure quality, you can't improve it

---

## 🏆 Best Practices

- ✅ Treat the LLM as **one component** in a larger system.
- ✅ Validate everything entering **and** leaving the model.
- ✅ Always return **sources** — trust in enterprise AI comes from traceability.
- ✅ Design for **cost and latency** from day one, not after launch.
- ✅ Log every request and response so you can debug and evaluate later.
- ✅ Start with the simplest thing that works, then add complexity only when measured
  data proves you need it.

---

## 📝 Assignment

Draw the architecture of an AI chatbot that answers questions from **uploaded PDF
documents**. Include these components:

1. User
2. Frontend
3. FastAPI backend
4. Document storage
5. Embedding model
6. Vector database
7. LLM
8. Safety layer
9. Response back to the user

You can draw it as a text diagram (like the ones above), or use a tool such as
[Excalidraw](https://excalidraw.com) or [draw.io](https://draw.io).

Save your answer in [`assignments/lesson-01/`](../../assignments/lesson-01/).

> Don't worry about making it perfect — we will refine it together. The point is to start
> thinking in **systems**, not in prompts.

---

## 📖 Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [What is RAG? (AWS)](https://aws.amazon.com/what-is/retrieval-augmented-generation/)

---

## ➡️ What's Next – Lesson 2

**How Large Language Models Actually Work**

We will cover:

- What an LLM really is
- Tokens and tokenization
- Embeddings (introduction)
- Transformers
- The attention mechanism (intuitive explanation)
- The context window
- Why ChatGPT "remembers" earlier messages
- Why LLMs hallucinate
- Differences between GPT, Claude, Llama, Qwen, Mistral, DeepSeek, and Gemma

By the end of the next lesson, you'll understand what happens **inside** an LLM instead
of treating it as a black box.

---

[⬅️ Back to course home](../../README.md)
