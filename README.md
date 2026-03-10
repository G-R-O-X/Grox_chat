# GROX Chat

Gemini Research Orchestration with minimaX -- Chat Only

[中文说明](README_CN.md)

A database-first, graph-orchestrated reasoning system for long-running technical discussion.

### Overview

GROX Chat is not a free-form group chat. It is a structured reasoning arena built around a persistent SQLite blackboard, with Gemini-led orchestration and MiniMax-driven debate turns.

- `Audience` plans the topic, opens each subtopic with a grounding brief, summarizes progress, and decides when to stop.
- The expert panel drives the actual reasoning: `Dreamer`, `Scientist`, `Engineer`, `Analyst`, `Critic`, and `Contrarian`.
- `Cat`, `Dog`, and `Tron` form the asynchronous validation layer.
- `Writer` verifies hard claims and turns stable findings into reusable facts.
- Retrieval is explicit: agents decide what to query, then run embedding, search, rerank, and injection before speaking.

### Execution Model

- Every speaking role performs local RAG before generating a message.
- `Round 1 / opening`: `Dreamer -> Scientist -> Engineer -> Analyst -> Critic -> Tron`
- `Round 2 / evidence`: `Dreamer -> Scientist -> Engineer -> Analyst -> Critic -> Contrarian -> Dog -> Cat -> Tron`
- `Round 3+ / debate`: same base roster, plus extra turns in fixed order `Tron -> Dog -> Cat`
- External web search is phase-gated: disabled in round 1, open to all speakers in round 2, and narrowed again in round 3+
- `Writer` runs at the end of every round and can persist verified facts into the fact store

### Architecture

```mermaid
flowchart LR
    User[User / Topic Input] --> Audience[Audience<br/>planning, grounding brief,<br/>summary, termination]
    Audience --> Arena[Subtopic Arena]
    Arena --> Writer[Writer<br/>fact verification]
    Writer --> Audience

    Memory[(SQLite Blackboard<br/>Topic / Plan / Subtopic / Message / Fact<br/>vec_facts / vec_messages)]

    Audience <--> Memory
    Arena <--> Memory
    Writer <--> Memory
```

### Subtopic Round Pipeline

```mermaid
flowchart TD
    Grounding[Audience grounding brief] --> Opening[Round 1 / opening<br/>Dreamer -> Scientist -> Engineer -> Analyst -> Critic -> Tron<br/>local RAG only]
    Opening --> Evidence[Round 2 / evidence<br/>Dreamer -> Scientist -> Engineer -> Analyst -> Critic -> Contrarian -> Dog -> Cat -> Tron<br/>local RAG + optional web search]
    Evidence --> Debate[Round 3+ / debate<br/>same base roster<br/>local RAG always on]
    Debate --> Extra[Extra turns if targeted<br/>Tron remediation -> Dog correction -> Cat expansion]
    Extra --> Writer[Writer fact pass<br/>every round]
    Writer --> Summary[Audience summary]
    Summary --> Eval[Audience termination check<br/>with summary-memory lookup]
    Eval -->|Continue| Next[Setup next round]
    Next --> PhaseShift{Next phase}
    PhaseShift -->|Round 2| Evidence
    PhaseShift -->|Round 3+| Debate
    Eval -->|Close| End[Close subtopic]
```

### Roles

- `Audience`: moderator, planner, summarizer, and topic-level controller.
- `Writer`: fact verifier and knowledge distiller; feeds the fact store.
- `Dreamer`: proposes new directions and hypotheses.
- `Scientist`: checks mechanism, theory, and internal validity.
- `Engineer`: converts ideas into buildable systems and concrete tactics.
- `Analyst`: contributes metrics, uncertainty estimates, and data framing.
- `Critic`: stress-tests claims and attacks weak reasoning.
- `Contrarian`: deliberately challenges the emerging consensus.
- `Cat`: rewards the strongest contribution with an extra turn.
- `Dog`: punishes the weakest claim with a forced re-verification turn.
- `Tron`: enforces forum laws around hallucination, bias, and logical safety.

### Retrieval and Memory

The system maintains two long-term memory lanes:

- `Fact RAG`: reusable, verified knowledge extracted by `Writer`.
- `Summary RAG`: historical summaries used by `Audience` to detect repetition, stalling, and semantic loops.

The intended retrieval path runs before every speaking turn:

1. Formulate a role-specific query.
2. Embed the query.
3. Retrieve candidate memories from the local vector store.
4. Rerank them with a cross-encoder.
5. Inject the top evidence into the next prompt.

### Repository Layout

- `src/grox_chat/`: orchestration, LLM clients, retrieval, persistence, prompts
- `tests/`: unit and integration tests
- `DESIGN.md`: full design description

### Quick Start

```bash
uv sync
cp .env.example .env
uv run python -c "from grox_chat.db import init_db; init_db()"
uv run python -m grox_chat.server
```

Create a topic from another shell:

```bash
uv run python -c "from grox_chat.api import create_topic; create_topic('Topic summary', 'Detailed topic prompt')"
```

### Smoke Test

Use a deliberately absurd but neutral topic:

```bash
uv run python -c "from grox_chat.api import create_topic; create_topic('From a workplace-practice perspective, should an employee enter with the left foot or the right foot?', 'From a workplace-practice perspective, should an employee enter with the left foot or the right foot?')"
```

Inspect the live state from another shell:

```bash
uv run python -c "import sqlite3; conn = sqlite3.connect('chatroom.db'); print(conn.execute('select count(*) from Subtopic').fetchone()[0], conn.execute('select count(*) from Message').fetchone()[0], conn.execute('select count(*) from Fact').fetchone()[0])"
```

Run tests:

```bash
uv run pytest -q
```
