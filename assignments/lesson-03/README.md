# Assignment – Lesson 3

## Task

Explain these concepts in your **own words**. If you can answer these clearly, you'll be
ahead of most candidates in AI engineering interviews.

### Core Questions

1. Why were RNNs replaced by Transformers?
2. What is self-attention?
3. Explain Query, Key, and Value using the library analogy.
4. Why do we need positional encoding?
5. What is the difference between encoder-only, decoder-only, and encoder–decoder
   Transformers?

### Bonus — The Follow-Ups That Separate Candidates

Anyone can recite the architecture. These are the questions that reveal whether you
actually understand it:

6. Why do we divide by `√d_k` in the attention formula?
7. What would happen if we removed the causal mask from GPT during training?
8. Your RAG system's latency tripled when you increased retrieved chunks from 5 to 15.
   Explain why, using attention complexity.

### Practical

9. Copy the self-attention code from section 6 of the lesson and run it. Change the
   embedding vector for `drank` and observe how the attention weights shift. Can you make
   `drank` attend most strongly to `milk`?
10. Set `mask=True` and confirm you get the triangular pattern. Explain in one sentence
    what each row means.

---

## How to Submit

Add your answers to this folder as either:

- `solution.md` — written answers, or
- `solution.ipynb` — a notebook, if you want to include the attention experiments

---

## Self-Check

You have genuinely understood this lesson if you can explain, without notes:

- [ ] Why parallelism — not accuracy — was the original breakthrough
- [ ] That Q, K, and V are three **learned projections of the same input**
- [ ] Why softmax saturation makes the `√d_k` scaling necessary
- [ ] Why BERT cannot generate text and GPT cannot do true bidirectional analysis
- [ ] Why attention is O(n²), and how that connects to the cost of long context
- [ ] What the KV cache is, and why it makes output tokens more expensive than input tokens
- [ ] At least two ways modern LLMs differ from the 2017 paper (RoPE, RMSNorm, GQA, SwiGLU)

If any of these are shaky, re-read that section before Lesson 4.

> 💡 **Strongest possible preparation:** watch Karpathy's
> [*Let's build GPT from scratch*](https://www.youtube.com/watch?v=kCc8FmEb1nY) and type
> the code yourself. Nothing cements this material faster than watching a Transformer you
> built begin to produce coherent text.
