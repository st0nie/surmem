# SurMem — Design Document

A surprise-gated, agent-agnostic memory framework.

## Core idea

Move Titans' surprise-driven memory update from the model-architecture layer to
the external memory layer, so it can be attached to **any** agent:

> **Store what is surprising, reinforce what repeats, forget what is routine.**

### Why surprise?

Humans don't remember every brick on their daily commute, but they do remember
the deer that suddenly appeared on the road. The value of a memory is
proportional to how unexpected its information is. Titans uses next-token
prediction error as its surprise signal inside the model; since we cannot
instrument an arbitrary agent's model internals, we approximate surprise with
**externally computable signals**:

| Titans (internal)                    | SurMem (external proxy)                                        |
| ------------------------------------ | -------------------------------------------------------------- |
| Next-token prediction error          | **Semantic novelty**: 1 − max cosine(new embedding, memories)  |
| Momentum term (sustained surprise)   | Accumulated novelty over a sliding time window                 |
| Decay gate (forgetting)              | Ebbinghaus-style strength decay + retrieval-driven reinforcement |

An optional LLM judge arbitrates the ambiguous middle zone (is this a new fact,
or does it contradict an old one?).

## Architecture

```
                        ┌─────────────────────────────────────────────┐
   Agent event stream ─▶│  SurpriseGate (write-side policy)            │
   (observe)            │                                              │
                        │   novelty = 1 − max cos(emb, memories)       │
                        │                                              │
                        │   novelty > tauAdd     → ADD (new memory)    │
                        │   sim     > dupSim     → REINFORCE (spacing) │
                        │   conflict zone        → UPDATE (supersede)  │
                        │   trivial / routine    → NOOP (discard)      │
                        └──────┬──────────────────────┬───────────────┘
                               ▼                      ▼
                    ┌──────────────────┐   ┌──────────────────────┐
                    │ Episodic store   │   │ Semantic store       │
                    │ raw, timestamped │──▶│ consolidated facts   │
                    │ decays fast      │consolidate│ decays slow   │
                    └──────────────────┘   └──────────────────────┘
                               │                      ▲
                               ▼                      │
                    ┌─────────────────────────────────┴───────────┐
                    │ Consolidator                                │
                    │ periodically clusters episodic memories and │
                    │ distills them into semantic facts           │
                    └─────────────────────────────────────────────┘

   Agent query ─────▶  Retriever: score = α·relevance + β·recency + γ·strength
                       strength = base · (1+ln(1+accessCount)) · exp(−λ·hours)
                       strength < threshold → forgotten (rescuable via REINFORCE)
```

## Cognitive layers

- **Episodic memory**: raw events with timestamps, recorded verbatim, decays fast
- **Semantic memory**: distilled by the consolidator from episodic clusters;
  stable, high-strength, decays ~10× slower
- **Procedural memory**: reserved interface — skills/workflows can be stored as
  special semantic memories

## Agent-agnostic design

1. **Two-call integration**: `await mem.observe(text)` (write path) and
   `await mem.recall(query)` (read path) plug into any agent loop.
2. **Pluggable embedder**: built-in zero-dependency `HashEmbedder` for demos;
   `OpenAIEmbedder` works against any OpenAI-compatible `/embeddings` endpoint
   (OpenAI, vLLM, Ollama, llama.cpp — including local embeddinggemma or
   Qwen3-Embedding GGUFs).
3. **Pluggable LLM judge / summarizer**: optional; without them the gate falls
   back to recency-wins conflict resolution and representative promotion.
4. **Replaceable write policy**: `SurpriseGate` is one policy; it can be swapped
   for an RL-trained policy later (the Memory-R1 direction).
5. **Replaceable storage**: in-memory by default; JSON persistence built in;
   SQLite / vector DB possible behind the same interface.

## Project layout

```
src/
  types.ts          # MemoryRecord / WriteVerdict / Kind
  embeddings.ts     # Embedder interface, HashEmbedder, OpenAIEmbedder
  gate.ts           # SurpriseGate: novelty + momentum + verdicts
  store.ts          # two-layer store, Ebbinghaus decay, JSON persistence
  retrieval.ts      # hybrid scoring (relevance + recency + strength)
  consolidation.ts  # episodic cluster -> semantic fact
  judge.ts          # OpenAIJudge (conflict arbitration), OpenAISummarizer
  index.ts          # SurpriseMemory facade + public exports
demo.ts             # full lifecycle demo (bun run demo.ts)
tests/              # bun test suite
.pi/extensions/surmem/   # pi coding-agent extension
```
