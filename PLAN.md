# GROX Chat Implementation Plan

## Goal

Evolve the baseline chatroom, not the conference system.

This plan implements the new `Skynet + Spectator + voting` design while
preserving the stable runtime work already completed for Gemini and MiniMax.

The work is intentionally split into small phases so the room stays runnable
throughout the migration.

## Phase 0: Stabilize the Runtime Surface

### Objective

Keep the current chatroom working while preparing for deeper role and governance changes.

### Deliverables

- preserve the current Gemini runtime fixes
  - warmup on worker start
  - project discovery retry and caching
  - in-process broker behavior
  - bounded concurrency
  - request coalescing for duplicate Gemini calls
- preserve MiniMax fallback behavior
- keep topic-scoped RAG and fact review working

### Acceptance

- current topics can still run end-to-end
- Gemini failure or absence still falls back cleanly to MiniMax where supported

## Phase 1: Introduce Agent Abstraction

### Objective

Stop scattering role logic, provider logic, and permission logic across graph nodes.

### Deliverables

- add a dedicated `AgentSpec` / `Agent` abstraction
- define explicit role classes:
  - `orchestrator`
  - `deliberator`
  - `special`
  - `npc`
- encode hard capabilities in metadata:
  - `can_vote`
  - `can_target`
  - `can_be_targeted`
  - `default_provider`
  - `default_strategy`
  - `default_tools`

### Initial roster

- `Skynet`
- deliberators:
  - `dreamer`
  - `scientist`
  - `engineer`
  - `analyst`
  - `critic`
  - `contrarian`
- special roles:
  - `cat`
  - `dog`
  - `tron`
  - `spectator`
- passive NPCs:
  - `writer`
  - `fact proposer`
  - `librarian`

### Acceptance

- role permissions are enforced by code, not only by prompts
- special roles cannot target other specials or NPCs

## Phase 2: Rename Audience to Skynet

### Objective

Replace the old orchestration identity cleanly.

### Deliverables

- rename prompt identity from `audience` to `Skynet`
- update topic orchestration prompts, summaries, and governance prompts
- keep the same functional responsibilities initially:
  - propose candidate subtopics
  - grounding brief
  - round summary
  - replan / close governance

### Acceptance

- no runtime references remain that depend on the old public `audience` persona
- summaries and governance messages are consistently attributed to `Skynet`

## Phase 3: Add Spectator

### Objective

Add a new special role that guides the next breakthrough turn without speaking as a normal debater.

### Behavior

- `Spectator` starts acting from round 2
- it selects exactly one ordinary deliberator
- its effect applies to the **next** round
- it injects a short focus prompt, for example:
  - `You feel that someone is watching you. Make this turn count.`
- if the targeted deliberator would otherwise not receive web-search access in that next round, grant a web-research boost

### Hard constraints

- `Spectator` may target only ordinary deliberators
- `Spectator` may repeatedly target the same deliberator across rounds
- `Spectator` is not a voting NPC and is not a passive NPC

### Acceptance

- `Spectator` never targets `cat`, `dog`, `tron`, `writer`, `librarian`, or `fact proposer`
- boosted turn behavior is visible and testable

## Phase 4: Replace Unilateral Subtopic Selection with Voting

### Objective

Subtopic admission should be determined by group voting rather than a single orchestrator decision.

### Rules

1. `Skynet` proposes `4` candidate subtopics
2. all non-NPC voting participants vote on each candidate
3. a candidate is selected only if it receives **more than two thirds** support
4. if fewer than `4` are selected, `Skynet` proposes more candidates to refill back to `4`
5. repeat up to `3` cycles by default
6. if after `3` cycles the number of selected subtopics is still `0`, close the topic
7. if at least `1` subtopic is selected, proceed normally

### Voting participants

- `Skynet`
- `dreamer`
- `scientist`
- `engineer`
- `analyst`
- `critic`
- `contrarian`
- `cat`
- `dog`
- `tron`
- `spectator`

### Non-voters

- `writer`
- `fact proposer`
- `librarian`

### Acceptance

- topic creation does not require selecting all `4` subtopics
- repeated `0-selected` outcomes eventually close the topic with an explanatory summary

## Phase 5: Replace Unilateral Continue / Replan Decisions with Voting

### Objective

Round continuation and replanning should become governance decisions.

### Deliverables

- vote on whether a subtopic should continue into another round
- vote on whether replanning is needed
- if replanning is needed, `Skynet` proposes new candidate subtopics and the room votes on them
- zero new subtopics is allowed when existing conclusions are sufficient

### Governance rule

- default acceptance threshold: more than two thirds support
- narrower rules should only exist if explicitly configured

### Acceptance

- no single speaking role can unilaterally continue or terminate a subtopic
- no single speaking role can unilaterally trigger or reject replanning

## Phase 6: Keep the Fact Pipeline but Enforce Role Boundaries

### Objective

Preserve the reviewed-fact architecture while making the role boundaries explicit.

### Deliverables

- keep the current pipeline:
  - `writer -> fact proposer -> librarian -> Skynet summary`
- ensure these passive NPCs never:
  - vote
  - receive special-role targeting
  - act as ordinary debate participants

### Acceptance

- facts remain reviewed before entry into `Fact`
- passive NPCs stay outside governance and targeting logic

## Phase 7: Unified LLM and Web-Search Broker

### Objective

Unify provider calls behind a single broker surface so the chatroom can control concurrency, retries, idempotency, and fallback in one place.

### Why

The room now depends on:

- Gemini orchestration calls
- MiniMax text generation
- MiniMax web-search loops
- future role-specific routing and boosts

Without a single gateway, rate limiting, duplicate work, and provider-specific failure handling will remain scattered.

### Requirements

- one broker layer for:
  - Gemini requests
  - MiniMax requests
  - web-search requests
- bounded concurrency
- duplicate in-flight request coalescing
- retry / backoff
- provider fallback
- optional per-role priority
- stable request identifiers for idempotent retries

### Suggested shape

- in-process first
- later optionally extract to a local sidecar/server if needed

### Acceptance

- identical orchestration requests do not fan out into duplicate upstream calls
- Gemini spikes do not create uncontrolled concurrent failures
- request routing becomes declarative from the `Agent` abstraction

## Phase 8: Tests and Migration Safety

### Objective

Make the governance rewrite safe to land incrementally.

### Required tests

- role-class permission tests
- special-role targeting matrix tests
- subtopic voting tests
- `0 selected` close-topic tests
- continue / replan voting tests
- `Spectator` boost tests
- broker idempotency and bounded-concurrency tests

### Migration rule

- land changes in order
- keep the old runtime path runnable until each new phase is covered by tests

## Deferred

Not part of this plan:

- conference mode
- Jedi Council
- panel clustering
- outlier panels
- candidate-subtopic conference banks
- persona libraries
- multi-room conference orchestration

Those belong to the separate upgraded system line, not to `grox_chat`.
