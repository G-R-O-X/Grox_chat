---
name: grox-code-review
description: Iterative code review with auto-selected expert agents (qa-expert, python-pro, cpp-pro, research-analyst), auto-fix, and convergence loop.
---

# Iterative Code Review Skill

## Overview

Multi-round code review that automatically selects relevant expert agents, triages findings, auto-fixes issues, and re-reviews until convergence or `max_rounds` is reached.

## Input Parameters

| Parameter | Required | Description |
|-----------|----------|-------------|
| `target` | Yes | File path(s), directory, or glob pattern to review |
| `max_rounds` | No | Maximum review iterations (default: 3) |
| `focus` | No | Review focus: `all` (default), `security`, `performance`, `correctness` |
| `auto_fix` | No | Auto-fix issues (default: true). If false, report only |
| `fix_level` | No | Minimum severity to auto-fix: `critical`, `major`, or `minor` (default, fix everything) |
| `prior_issues` | No | Known issues from a previous session (free-text or structured table). Injected into Phase 2 as prior context so agents skip re-discovering them and focus on verifying fixes or finding new issues |

---

## Phase 1: Analyze Target & Select Agents

### Step 1: Read Target Files

Read all target files. If `target` is a directory, collect source files recursively (respect `.gitignore`). Skip binaries, lock files, and generated files.

### Step 2: Classify Codebase

Analyze the collected files to determine:

1. **Languages present**: Check file extensions
   - `.py` → Python
   - `.cpp`, `.cc`, `.cxx`, `.h`, `.hpp` → C++
   - Other extensions → note but no specialist agent
2. **Algorithmic complexity**: Scan for signals of non-trivial algorithms
   - Mathematical operations: `numpy`, `scipy`, `torch`, matrix operations, numerical methods
   - Complex data structures: custom trees, graphs, heaps, union-find
   - Algorithm keywords: `dynamic programming`, `convex`, `gradient`, `convergence`, `eigenvalue`, `decomposition`
   - Research-adjacent patterns: paper references in comments (`[Author et al.]`, `arXiv`), academic-style variable names (`\alpha`, `epsilon`)

### Step 3: Select Agents

Always include: **qa-expert** (universal, catches logic/edge-case/resource issues)

Conditionally include:

| Agent | Condition | Rationale |
|-------|-----------|-----------|
| `python-pro` | Any `.py` files in target | Python-specific anti-patterns, type safety, async correctness |
| `cpp-pro` | Any `.cpp/.cc/.h/.hpp` files in target | C++ memory safety, RAII, template issues, UB |
| `research-analyst` | Algorithmic complexity signals detected (Step 2 above) | Validates algorithmic correctness, numerical stability, mathematical soundness |

It is valid to select only `qa-expert` (e.g., for a shell script or config file). It is also valid to select all four (e.g., a Python+C++ project with numerical algorithms).

### Output

```
Selected agents: [qa-expert, python-pro, research-analyst]
Reason: Python files detected, numpy/scipy usage with convergence checks
```

---

## Phase 2: Parallel Expert Review

Launch all selected agents **in parallel** (single message, multiple Agent tool calls). Each agent reviews the same files from its perspective.

### Agent Prompts

For each selected agent, use the corresponding prompt below. Replace `{files_content}` with the actual file contents, `{focus}` with the review focus, and `{prior_issues}` with issues from prior rounds.

**Round 1 with `prior_issues` input:** If the user provided `prior_issues`, inject them as prior context on round 1 (same format as rounds 2+). This lets the skill resume from a previous session — agents will verify whether those issues still exist, check if fixes are correct, and look for new issues only.

**Round 1 without `prior_issues`:** Leave `{prior_issues}` empty.

#### qa-expert

```
Review the following code for quality and correctness defects.

Focus: {focus}

{prior_issues}

Files to review:
{files_content}

Check for:
- Edge cases and boundary conditions not handled
- Error handling gaps (uncaught exceptions, missing validation)
- Race conditions and concurrency issues
- Resource leaks (unclosed files, connections, locks)
- Logic errors (off-by-one, wrong operator, inverted condition)
- Input validation gaps at system boundaries
- Inconsistent state handling

DO NOT flag: style preferences, missing docstrings on clear code, hypothetical future requirements.

For each issue, output exactly this JSON format:
[
  {
    "id": "QA-1",
    "file": "path/to/file",
    "line": 42,
    "severity": "critical|major|minor",
    "category": "error_handling|logic|concurrency|validation|resource_leak|edge_case",
    "description": "What the issue is",
    "suggestion": "Concrete fix (show code if possible)"
  }
]

If no issues found, output: []
```

#### python-pro

```
Review the following Python code as a senior Python engineer.

IMPORTANT — Python environment rules:
- If a `pyproject.toml` with `[tool.uv]` or a `.venv` managed by uv exists, use `uv run` to invoke all Python tools (e.g. `uv run pytest`, `uv run python -m py_compile`). NEVER use bare `python3`, `pip`, or `pytest` directly.
- If no uv environment but conda is active, use `conda run` or the conda env's Python.
- Only fall back to system `python3` / `pip` if neither uv nor conda is available.
- NEVER install packages into the system Python. If a dependency is missing, report it instead of installing.

Focus: {focus}

{prior_issues}

Files to review:
{files_content}

Check for:
- Anti-patterns (mutable default args, bare except, global state abuse)
- Type safety issues (wrong types, missing narrowing, Any leakage)
- API misuse (wrong method, deprecated usage, incorrect arguments)
- Async correctness (blocking in async, missing await, event loop issues)
- Performance anti-patterns (N+1, unnecessary copies, quadratic loops)
- Security vulnerabilities (injection, path traversal, unsafe deserialization, eval/exec)
- Resource management (missing context managers, unclosed resources)

DO NOT flag: personal preference with no correctness impact, missing optional type annotations on internal helpers, code that is clear without comments.

For each issue, output exactly this JSON format:
[
  {
    "id": "PY-1",
    "file": "path/to/file.py",
    "line": 42,
    "severity": "critical|major|minor",
    "category": "anti_pattern|type_safety|api_misuse|async|performance|security|resource",
    "description": "What the issue is",
    "suggestion": "Concrete fix (show code if possible)"
  }
]

If no issues found, output: []
```

#### cpp-pro

```
Review the following C++ code as a senior C++ engineer.

IMPORTANT — Build environment rules:
- If a `pyproject.toml` with `[tool.uv]` exists, use `uv run` for any Python-related commands (e.g. `uv run python`, `uv run pytest`).
- NEVER install packages into the system Python or use bare `pip`.
- For C++ builds, prefer the project's existing build system (CMakeLists.txt, Makefile).

Focus: {focus}

{prior_issues}

Files to review:
{files_content}

Check for:
- Memory safety (use-after-free, double-free, buffer overflow, dangling pointers/references)
- RAII violations (raw new/delete, missing smart pointers, resource leaks)
- Undefined behavior (signed overflow, null deref, uninitialized reads, aliasing violations)
- Template/constexpr correctness (SFINAE errors, concept violations, ODR issues)
- Move semantics misuse (moved-from access, unnecessary copies, missing noexcept)
- Concurrency issues (data races, lock ordering, deadlocks)
- API misuse (incorrect STL usage, deprecated features, platform-specific assumptions)

DO NOT flag: style preferences, brace placement, naming conventions unless they cause bugs.

For each issue, output exactly this JSON format:
[
  {
    "id": "CPP-1",
    "file": "path/to/file.cpp",
    "line": 42,
    "severity": "critical|major|minor",
    "category": "memory|raii|ub|template|move|concurrency|api_misuse",
    "description": "What the issue is",
    "suggestion": "Concrete fix (show code if possible)"
  }
]

If no issues found, output: []
```

#### research-analyst

```
Review the following code for algorithmic and mathematical correctness.

Focus: {focus}

{prior_issues}

Files to review:
{files_content}

Check for:
- Numerical stability (catastrophic cancellation, loss of significance, overflow/underflow)
- Algorithm correctness (wrong complexity, incorrect invariants, missing termination proof)
- Convergence issues (wrong stopping criteria, missing divergence checks)
- Mathematical errors (wrong formula, index errors in matrix ops, incorrect gradients)
- Statistical issues (wrong distribution assumptions, biased estimators, incorrect p-values)
- Approximation errors (wrong error bounds, missing tolerance parameters)
- Edge cases in numerical code (division by zero, log of zero, empty inputs to aggregations)

DO NOT flag: code style, non-algorithmic logic, general software engineering issues (other agents handle those).

For each issue, output exactly this JSON format:
[
  {
    "id": "RA-1",
    "file": "path/to/file",
    "line": 42,
    "severity": "critical|major|minor",
    "category": "numerical_stability|algorithm|convergence|math_error|statistical|approximation|edge_case",
    "description": "What the issue is (include the mathematical reasoning)",
    "suggestion": "Concrete fix (show corrected formula/code)"
  }
]

If no issues found, output: []
```

### Prior Issues Context

On rounds 2+, prepend this to each agent prompt:

```
PRIOR ROUND CONTEXT:
The following issues were found and fixed in previous rounds. Do NOT re-report these.
Instead, check whether the fixes are correct and look for NEW issues only.

Previously found and fixed:
{list of fixed issues with before/after code}

Previously found but NOT fixed (still open):
{list of unfixed issues}
```

---

## Phase 3: Triage & Deduplicate

### Step 1: Merge

Combine all agent issue lists into one list.

### Step 2: Deduplicate

Two issues are duplicates if they reference the same file + overlapping line range (within 3 lines) + same semantic category. Keep the one with higher severity and more detailed suggestion. Note which agents agreed (consensus = higher confidence).

### Step 3: Classify

```
issues = {
  "critical": [...],   # Must fix: crashes, security holes, data loss, UB
  "major": [...],      # Should fix: logic errors, resource leaks, numerical instability
  "minor": [...],      # Nice to fix: anti-patterns, mild performance, style-adjacent
}
```

---

## Phase 4: Auto-Fix (if `auto_fix=true`)

Fix all issues at or above `fix_level` severity. Default `fix_level=minor` means all issues are auto-fixed. Set `fix_level=major` to only fix critical+major, or `fix_level=critical` to only fix critical.

### Step 1: Apply Fixes

For each issue at or above `fix_level`, ordered by file then line number (top to bottom):

1. Read the file around the issue location (within 20 lines for context)
2. Apply the suggested fix via Edit tool (minimal, targeted edit)
3. If multiple issues overlap in the same region, consider them together

### Step 2: Verify Syntax

After all fixes in a file:
- Python: `uv run python -m py_compile {file}` (fall back to `python -m py_compile {file}` only if uv is unavailable)
- C++: `g++ -fsyntax-only -std=c++20 {file}` (or `cmake --build . --target {target}` if CMakeLists.txt exists)
- Other: skip

If a fix breaks syntax, **revert it** and mark as `needs_manual_fix`.

### Step 3: Track

```
applied_fixes = [
    {"issue_id": "QA-1", "file": "...", "status": "fixed"},
    {"issue_id": "PY-2", "file": "...", "status": "reverted_syntax_error"},
]
```

---

## Phase 5: Convergence Gate

### Count Remaining

```
remaining = [issues at or above fix_level severity that are NOT status=fixed]
```

### Exit Conditions (any one triggers exit)

1. **Clean**: `len(remaining) == 0` — converged, all fixable issues fixed
2. **Max rounds**: `round >= max_rounds` — stop, report remaining issues to user
3. **Oscillation**: Round N finds issues identical to those "fixed" in round N-1 — conflicting opinions, stop and report to user for judgment

### Continue Condition

`len(remaining) > 0 AND round < max_rounds AND no oscillation` — go back to **Phase 2** with:
- Only files that were modified in Phase 4 (unchanged files don't need re-review)
- Prior issues context injected into agent prompts

---

## Phase 6: Final Report

Print directly to user (do NOT write to file unless asked):

```markdown
# Code Review Report

**Target:** {target}
**Rounds:** {total_rounds} / {max_rounds}
**Agents used:** {agent_list} ({reasons})
**Result:** {Converged | Stopped at max_rounds | Oscillation detected}

## Summary
| Severity | Found | Fixed | Remaining |
|----------|-------|-------|-----------|
| Critical | X | X | 0 |
| Major | Y | Y-1 | 1 |
| Minor | Z | 0 | Z |

## Issues

### Critical
| ID | File | Line | Category | Description | Status |
|----|------|------|----------|-------------|--------|
| QA-1 | foo.py | 42 | error_handling | Uncaught ValueError | Fixed |

### Major
...

### Minor
...

## Fix Details

### QA-1: Uncaught ValueError (foo.py:42)
**Before:**
(original code)
**After:**
(fixed code)

## Round History
- R1: 5 issues (2C, 2M, 1m), agents: [qa-expert, python-pro] -> fixed 4
- R2: 0 new issues -> converged
```

---

## Example Usage

### Basic (auto-selects agents, 3 rounds max)
```
/grox-code-review
target: src/grox_chat/server.py
```

### Limit to 2 rounds
```
/grox-code-review
target: src/grox_chat/rag.py
max_rounds: 2
```

### Only fix critical and major (skip minor)
```
/grox-code-review
target: src/grox_chat/broker.py
fix_level: major
```

### Report only, no auto-fix
```
/grox-code-review
target: src/grox_chat/prompts.py
auto_fix: false
```

### Resume from previous session with known issues
```
/grox-code-review
target: src/grox_chat/server.py
prior_issues: |
  QA-2 (major): resp.json() can throw JSONDecodeError, uncaught
  PY-3 (major): HTTPStatusError handler discards response body
```

### Directory with focus
```
/grox-code-review
target: src/grox_chat/
focus: security
max_rounds: 2
```
