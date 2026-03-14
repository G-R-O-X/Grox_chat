# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GROX Chat is a multi-agent deliberation system where AI agents debate topics through structured rounds. It uses LangGraph for orchestration, SQLite (with sqlite-vec) for persistence, and a dual-LLM broker (Gemini primary, MiniMax fallback) with web search.

## Commands

```bash
uv sync                          # Install dependencies
uv run pytest -q                 # Run all tests
uv run pytest tests/test_foo.py -q  # Run single test file
uv run ruff check src tests      # Lint
uv run python -m grox_chat.server   # Start server
```

Tests use `TESTING=1` env var to switch to `test_chatroom.db`. API keys go in `.env` (see `.env.example`).

## Architecture

### Three-Layer Design

1. **Topic Orchestration** (`master_graph.py`) — LangGraph state machine managing topic lifecycle: plan generation → subtopic voting (>2/3 majority) → subtopic execution → replan or close. Uses Gemini for orchestration calls.

2. **Subtopic Arena** (`server.py`) — Round-based debate loop using a dispatcher pattern with `TurnSpec` queue. Rounds progress: OPENING (local RAG only) → EVIDENCE (web search allowed, special roles act) → DEBATE. Each agent turn is a graph node routed by the dispatcher.

3. **Fact Pipeline** (NPC agents, no voting rights) — Writer extracts candidates → Fact Proposer structures them → Librarian reviews (accept/soften/reject). Only reviewed facts enter the permanent `Fact` table and become RAG-visible. Parsing logic in `writer_processor.py` and `librarian_processor.py`.

### Key Modules

- **`api.py`** — Public API surface (CRUD for topics, plans, subtopics, messages, facts)
- **`db.py`** — SQLite schema, sqlite-vec dense embeddings, FTS5 indexes, context-manager pattern
- **`graph.py`** — LangGraph state types (`ChatState`, `TurnSpec`) and dispatcher logic
- **`broker.py`** — Unified LLM broker with provider fallback, bounded concurrency (semaphore=8), web search with 30-day cache, request coalescing
- **`agents.py`** — Agent specs with explicit permission flags (`can_vote`, `can_target`, `can_be_targeted`), role classification into Orchestrator/Deliberator/Special/NPC classes
- **`rag.py`** — Topic-scoped RAG: dense retrieval + FTS5 lexical + cross-encoder reranking (threshold 0.3)
- **`prompts.py`** — Role-specific system prompts (Skynet, Writer, Fact Proposer, Librarian, Deliberators)

### Agent Role Classes

| Class | Examples | Votes | Notes |
|-------|----------|-------|-------|
| Orchestrator | skynet | Yes | Proposes subtopics, writes summaries |
| Deliberator | dreamer, scientist, engineer, analyst, critic, contrarian | Yes | Normal debate speakers, targetable |
| Special | cat, dog, tron, spectator | Yes | Interventions; can only target deliberators |
| NPC | writer, fact_proposer, librarian | No | Fact pipeline only, not targetable |

### Database

SQLite with sqlite-vec for vector storage. Core tables: Topic, Plan, Subtopic, Message, FactCandidate, Fact, ClaimCandidate, WebEvidence, VoteRecord. Dense embeddings on Message and Fact tables. FTS5 full-text indexes on facts, messages, claims, and web evidence.

### Citation Format

The system enforces citation markers: `[F...]` for facts, `[C...]` for claims, `[W...]` for web evidence. RAG assembly strips and re-injects citations.
