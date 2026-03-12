# GROX Chat Governance, Summary, and Voting Redesign Plan

## Purpose

This plan replaces the older implementation checklist with a focused redesign plan for:

- round summaries
- subtopic termination governance
- structured voting outputs
- aggregation logic
- governance observability
- alignment between design intent and runtime behavior

The plan is based on two external prompt-design critiques plus direct inspection of the current runtime.


## Core Problem

The main failure mode is premature closure.

The system currently has several related weaknesses:

- summaries can still narratively smooth over unresolved disagreement
- termination prompts are still too soft and too textual
- voters can close a subtopic without explicitly naming what remains unresolved
- stage labels like `weak` / `medium` / `strong` are not enough by themselves
- admitted facts may remain conditional or softened, but this does not yet constrain closure strongly enough
- the runtime did not previously respect the intended round cadence for termination voting

This produces a predictable bad outcome:

- the room appears rhetorically converged
- but the actual recommendation, evidence base, or governing metric is still unstable
- and the room votes to close too early


## Design Principles

The redesign will follow these principles:

- summary is state compression, not closure judgment
- governance must be structurally conservative against premature closure
- unresolved central branches must be made explicit before any close vote
- recent framing shifts must be treated as evidence of instability
- structured booleans and categorical fields are better than free-form numeric readiness scores
- the code should prefer deterministic safeguards over hoping the model "understands the spirit"


## Final Design Goals

The final system should behave like this:

- round 1 is opening only: establish initial positions, no termination vote
- round 2 is evidence gathering: challenge, refine, and collect facts, no termination vote
- round 3 and later are eligible for termination governance
- every round summary separates:
  - macro trajectory
  - current consensus
  - live disagreements
  - evidence gaps
  - participant deltas
- summaries must explicitly report Librarian rulings and the caveats attached to admitted facts
- termination voters must explicitly identify:
  - the main unresolved branch
  - whether the recommendation or governing metric changed recently
  - whether open questions are central or peripheral
  - whether the core recommendation still depends on conditional or weakly validated facts
  - whether a newly introduced framework has not yet been stress-tested
- closure should not be determined by a plain yes/no majority alone
- the aggregation logic must block close when a small but meaningful minority identifies a central unresolved blocker
- round-stage guidance should be written as burden-of-proof rules, not vague threshold words
- round 10 remains a deterministic forced close in code, not an LLM decision


## MiniMax-First Execution Constraints

The final architecture must be designed for the model that actually carries most of the live workload today:

- MiniMax is the primary runtime model
- Gemini may be disabled or degraded
- governance prompts therefore must be optimized for MiniMax stability first, not for ideal richness

This changes the implementation strategy in several concrete ways.

### Schema simplicity beats expressive richness

The first production version of structured governance should use a flat schema with short categorical fields.

Preferred initial shape:

```json
{
  "main_branch": "...",
  "centrality": "central|mixed|peripheral|none",
  "recent_shift": "yes|no|unclear",
  "conditional_support": "yes|no",
  "untested_novelty": "yes|no",
  "vote": "continue|close",
  "override_reason": "... or null"
}
```

The richer nested schema remains the target design language, but the MiniMax-safe rollout should start flat and only add structure after runtime validation.

### Keep rationale short or omit it in the first rollout

Long governance rationales increase the chance of:

- malformed JSON
- `<think>` leakage
- prompt drift
- contradictory fields

The first MiniMax-safe rollout should either:

- omit `rationale` entirely, or
- constrain it to one short sentence

### Summary structure must also be MiniMax-safe

The final summary architecture still separates consensus, blockers, and evidence gaps.

But the first rollout should use short, hard sections with minimal narrative blending:

- `TRAJECTORY`
- `CONSENSUS`
- `BLOCKERS`
- `EVIDENCE GAPS`
- `AGENT DELTAS`

This is a MiniMax-safe compression format.

Do not start with long prose and then expect the governance voter to recover the true state later.

### Aggregation must ignore free-text phrasing

MiniMax is not reliable enough for blocker aggregation based on branch-name string matching.

The aggregation layer should only trust:

- categorical fields
- booleans
- normalized vote labels

The text field like `main_branch` is for audit only, not for hard logic.

### Normalize and repair are mandatory, not optional

Every structured governance output should go through:

- JSON normalization
- schema validation
- MiniMax repair on failure

This applies to:

- round summaries when strict structure is required
- termination votes
- later admission / replan structured votes if they are upgraded

### Rollout must happen in two layers

The redesign should distinguish between:

- `MiniMax-safe governance`
- `Richer governance when Gemini is enabled`

The MiniMax-safe version is the default production path.

The richer version can only be enabled once:

- the MiniMax-safe schema is stable
- repair frequency is acceptable
- aggregation logic is proven on real runtime traces


## Scope

This redesign covers:

- subtopic round summaries
- subtopic continuation / close voting
- vote logging and auditability
- eventual alignment of subtopic-admission and replan votes with the same governance philosophy

This redesign does not cover:

- conference mode
- multi-room orchestration
- non-chatroom product pivots


## Current State

The runtime already has some of the necessary foundation:

- voting-based governance exists
- the writer -> fact proposer -> librarian -> skynet summary pipeline exists
- per-agent raw vote responses are now visible in logs
- round 1 / round 2 termination voting has been disabled in code
- the summary prompt already asks for:
  - agent positions
  - synthesis
  - open questions
- the termination prompt already asks the model to look for:
  - central open questions
  - recent framing changes
  - conditional or softened facts
  - untested novelty

That is a useful transitional state, but not the final architecture.


## Final Summary Design

The final summary should no longer be a single blended narrative with only one `SYNTHESIS` section carrying all governance signal.

### Required sections

The target summary structure is:

1. `MACRO TRAJECTORY & SHIFTS`
2. `CONSENSUS & LIBRARIAN RULINGS`
3. `LIVE DISAGREEMENTS (THE BLOCKERS)`
4. `EVIDENCE GAPS & OPEN QUESTIONS`
5. `AGENT POSITIONS (APPENDIX)`

### Summary rules

- `MACRO TRAJECTORY & SHIFTS` must state whether the room materially changed framing, governing metric, or recommendation during this round
- `CONSENSUS & LIBRARIAN RULINGS` must include the strongest current agreement plus all caveats attached to the relevant admitted facts
- `LIVE DISAGREEMENTS` must state the main branch or branches still blocking closure
- `EVIDENCE GAPS` must distinguish central gaps from peripheral gaps
- `AGENT POSITIONS` should emphasize deltas:
  - what changed
  - what was conceded
  - what new attack or correction was introduced

### Explicit non-goal

The summary should not declare whether the room is ready to close.

Closure readiness belongs to the governance vote, not to Skynet's summary.


## Final Termination Vote Design

The final termination vote should be structured, auditable, and more conservative than the current yes/no-only scheme.

### Governance posture

Termination is not a "did we talk enough?" vote.

It is a structured test of whether:

- the recommendation is stable
- the remaining disputes are peripheral
- the evidence base is strong enough
- the room is no longer changing its core framing

### Final vote output schema

The target schema is:

```json
{
  "state_analysis": {
    "main_unresolved_branch": "...",
    "open_questions_centrality": "central | mixed | peripheral | none",
    "recommendation_changed_recently": "yes | no | unclear",
    "what_changed_recently": "... or null"
  },
  "premature_closure_checklist": {
    "has_central_unresolved_branch": true,
    "has_recent_volatility": true,
    "central_claim_support_is_conditional": true,
    "has_untested_novelty": true
  },
  "rationale": "...",
  "vote": "continue | close",
  "override_reason": "... or null"
}
```

### Why this schema

- it forces state extraction before the final vote token
- it records why a vote happened
- it avoids vague numeric scoring
- it makes aggregation auditable
- it separates blockers from final decision

### Important refinement over external suggestions

For live rollout, use the MiniMax-safe flat schema first.

The nested schema above is the conceptual target, not the required first deployment format.

The field should not be `has_conditional_evidence` in the broad sense.

The final design should use a narrower concept:

- `central_claim_support_is_conditional`

This prevents normal caveats in admitted facts from blocking closure forever.


## Final Stage Guidance

The final stage rules should be written as burden-of-proof instructions.

### Round 3

Early exploration stage.

- bias strongly toward `continue`
- if any central blocker is true, vote `continue`
- vote `close` only if the subtopic is trivially resolved and the summary shows no live disagreement

### Rounds 4-6

Refinement stage.

- balanced burden of proof
- vote `continue` if the room still has:
  - recent volatility
  - a central unresolved branch
  - a central evidence gap
- vote `close` only when the remaining disagreement is peripheral or repetitive

### Rounds 7-9

Convergence stage.

- bias toward `close`
- vote `continue` only when a specific severe blocker remains
- peripheral questions should not keep the room open

### Round 10

Deterministic forced close in code.

No LLM decision should be used here.


## Aggregation Design

The final aggregation logic should no longer treat governance as plain majority yes/no.

### Current weakness

A simple majority can close the subtopic even when a minority has correctly identified a specific central unresolved branch.

### Final aggregation target

The system should close a subtopic only if all of the following hold:

- close-vote support clears the configured threshold
- fewer than the configured blocker threshold of voters identify a central unresolved branch
- recent volatility is not being flagged by a meaningful minority
- the recommendation is not still supported mainly by conditional or softened facts
- no strong blocker survives without a valid override reason

### Default intended policy

The target default is:

- close threshold: greater than or equal to a configured supermajority
- blocker threshold: at least two independent blocker flags should be enough to prevent close at early and middle stages
- later rounds may relax this slightly, but never to a plain majority-only rule

### Important implementation note

Aggregation should use structured categorical fields and booleans.

It should not depend on exact string matching of free-text branch names, because different agents will describe the same blocker differently.

For MiniMax-first rollout, this is a hard rule, not just a preference.


## Governance Prompting Model

The final governance mode should preserve each role's lens without letting persona noise degrade structured voting.

### Planned policy

- voters still keep their role perspective:
  - critic should be hard to convince
  - scientist should demand evidence stability
  - contrarian should pressure weak consensus
  - engineer should care about implementation realism
- but governance voting should use a dedicated structured voting contract
- cat / dog / tron / spectator should be evaluated carefully:
  - either keep them as voters with the same structured schema
  - or narrow the governance voter set if they add more noise than signal

This needs runtime validation before becoming permanent policy.


## Observability and Storage

The final system needs more than raw log lines.

### Minimum target

- keep logging the full raw vote response
- log the parsed structured vote object
- log the final blocker counts and close decision

### Preferred target

Add durable storage for governance decisions, for example:

- per-round vote artifacts
- parsed governance checklist fields
- final aggregation result
- stage guidance used

This can be a new DB table or a structured event log.


## Implementation Plan

### Phase 1: Summary hardening

- replace the blended summary design with explicit sectioned state compression
- incorporate Librarian rulings directly into the consensus section
- move per-agent positions to appendix style
- require delta-focused participant bullets instead of static restatements

### Phase 2: Termination schema redesign

- replace yes/no-only governance output with structured JSON
- switch final vote tokens from `yes/no` to `continue/close`
- begin with the MiniMax-safe flat schema
- require unresolved-branch and volatility extraction before vote
- keep rationale minimal or disabled in the first rollout

### Phase 3: Aggregation redesign

- extend vote parsing beyond booleans
- aggregate blocker fields explicitly
- block closure when central unresolved signals persist
- retain round-10 forced close
- do not use free-text blocker names in aggregation logic

### Phase 4: Governance prompt cleanup

- rewrite round-stage guidance as burden-of-proof instructions
- make the termination prompt explicitly conservative against premature closure
- ensure the summary prompt never signals closure readiness
- shorten and harden summary sections for MiniMax-first execution

### Phase 5: Observability

- persist or log structured vote state
- expose final tallies and blockers cleanly for runtime inspection
- track JSON repair frequency so schema complexity is driven by runtime evidence, not theory

### Phase 6: Admission and replan alignment

- revisit candidate-subtopic admission prompts
- decide whether they should remain simple or gain a lighter structured checklist
- align replan voting with the same governance philosophy without overcomplicating it
- only upgrade these flows after subtopic termination governance is stable under MiniMax


## Testing Plan

The redesign must be covered by regression tests.

### Summary tests

- summary prompt contains the required sections
- summary prompt requires disagreement and evidence-gap separation
- degraded summary still preserves structural headers

### Termination tests

- round 1 and round 2 skip termination voting
- round 3 and later run structured governance
- vote parsing handles nested JSON fields
- close is blocked when central unresolved branches are flagged
- close is blocked when recent volatility is flagged
- close is blocked when key support facts remain conditional
- round 10 forces close regardless of model output

### Aggregation tests

- a supermajority close vote can still be blocked by structured blockers
- branch-name wording differences do not break blocker aggregation
- override reasons are required when a close vote contradicts default blocker logic

### Runtime tests

- live topic execution no longer closes subtopics during rounds 1-2
- summaries preserve unresolved disagreement instead of smoothing it away
- logs and stored artifacts expose why close / continue happened


## Success Criteria

This redesign is successful when:

- the room does not terminate during opening or evidence-gathering rounds
- summaries no longer blur core disagreement into caveated consensus
- voters can explain why they want to continue or close
- closure decisions become auditable after the fact
- late-stage convergence still happens without the room stalling forever
- the system closes subtopics because blockers are truly gone, not because the rhetoric sounds mature


## Immediate Next Steps

1. Keep the current round-1/2 termination gate.
2. Finish the sectioned summary redesign.
3. Replace boolean-only termination outputs with structured vote JSON.
4. Redesign aggregation to use blocker fields, not only close-vote counts.
5. Add vote observability before widening the change to admission and replanning flows.
