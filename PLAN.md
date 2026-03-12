# GROX Chat Implementation Plan

## Current Baseline

`grox_chat` is now the upgraded base chatroom, not the conference-mode branch.

The current implemented baseline includes:

- `Skynet` as the public orchestration identity
- explicit role classes:
  - `orchestrator`
  - `deliberator`
  - `special`
  - `npc`
- `Spectator` as a next-round focus and web-boost role
- voting-based subtopic admission
- voting-based round continuation and replanning
- a reviewed fact pipeline:
  - `writer -> fact proposer -> librarian -> Skynet summary`
- one in-process broker for Gemini, MiniMax, and web-search execution

## Completed Work

The following migration phases are already implemented in the runtime:

1. runtime stabilization and fallback preservation
2. `AgentSpec` / `Agent` abstraction
3. `Skynet` rename
4. `Spectator`
5. initial subtopic voting
6. round continuation and replanning voting
7. role-boundary enforcement for passive NPCs
8. unified broker routing
9. regression hardening around brokered voting and RAG fallback

## Remaining Work

The remaining work is now narrower and operational:

### 1. Runtime smoke validation

- validate live topic execution against the current baseline
- validate zero-selected retry behavior
- validate round continuation voting
- validate replan voting
- validate Gemini degradation into MiniMax fallback
- validate `Spectator` next-round boost behavior under real runtime conditions

### 2. Next feature selection

After the baseline is fully re-synced and smoke-validated, choose the next feature direction from the real current system state. Candidates include:

- better vote observability and audit output
- a cleaner public `Agent` API surface
- broker/server operational tooling
- runtime administration and smoke tooling

## Guidance

- `DESIGN.md` remains the product source of truth.
- `.plan/` is the local execution workspace and should track only the remaining work.
- No conference-mode features should be implemented in this repository.

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
