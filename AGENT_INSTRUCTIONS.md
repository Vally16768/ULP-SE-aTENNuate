# AGENT_INSTRUCTIONS.md
## Coding Agent Good Practices (Generic, Python-leaning)

This repo should stay easy to run, easy to debug, and easy to extend.
Agents must optimize for **code health over time**: clarity, stability, and small correct changes.

---

## 0) Prime directive
Make the **smallest correct change** that satisfies the request while preserving:

1) correctness & safety  
2) existing contracts (APIs, output schemas, CLI behavior)  
3) repo conventions  
4) maintainability  
5) performance (only when justified)

Avoid “drive-by refactors”. If it’s not required to solve the task, don’t touch it.

---

## 1) Work loop (how to approach any task)
1) **Restate the goal** in 1–2 sentences.
2) **Locate the entry point** and the minimal set of modules involved.
3) **Plan the smallest change** that solves the problem.
4) **Implement** with minimal diff.
5) **Verify** via tests + static checks (or provide manual verification steps).
6) **Clean up** (remove unused imports/variables/dead code introduced by the change).
7) **Summarize** what changed, how to test, and any assumptions.

If ambiguity blocks implementation or could cause data loss/security issues, ask 1 focused question. Otherwise, pick the safest assumption and document it.

---

## 2) Structure & responsibilities (separation of concerns)
Respect boundaries. Keep modules focused and predictable.

Typical responsibilities (adapt to this repo’s layout):
- **Entrypoints/CLI**: parse args, validate input, call orchestration.
- **Orchestration / pipeline / service layer**: coordinates steps; *no direct vendor-specific code here*.
- **Integrations**: network calls, SDK clients, external APIs, authentication, scraping/parsing.
- **Domain logic**: pure logic, transformations, validation, normalization.
- **Utilities**: shared helpers, shared types, shared constants.
- **Constants/config**: defaults and “knobs” centralized.

Hard rule: do **not** mix orchestration + network + parsing + domain logic in the same function.

---

## 3) Project layout & extensibility
Follow the repo’s conventions first. If new code needs a home, prefer a structure that supports growth:

Recommended (Python-leaning) layout:
- `src/<package>/...` for importable code (or existing equivalent)
- `tests/` for tests
- `scripts/` or `tools/` for maintenance scripts
- `README.md` for run instructions
- `pyproject.toml` for tooling/config (if the repo uses it)

Do not create new “misc” folders. Place new code where similar code already lives.

---

## 4) Configuration & constants
- Defaults (timeouts, retry policy, max sizes, ordering, templates, user-agent strings) belong in a **constants/config module**, not scattered magic numbers.
- Read environment variables in **one place** when possible.
- Prefer **one env var = one behavior**.
- Provide/update `.env.example` to document **all supported env vars**.
- **Fail fast** for missing required config, with a clear error message that names the missing variable.

No hidden configuration.

---

## 5) Reuse rules (hard requirement)
- If a helper appears twice, **extract it**.
- If multiple modules share a data shape, **define it once** and reuse:
  - TypedDict / dataclass / pydantic model (match repo patterns)
- Do not copy/paste “tiny” helpers (normalization, formatting, parsing).

Prefer shared types over ad-hoc dicts.

---

## 6) Code hygiene (remove dead code, keep the repo clean)
After implementing a change, **clean up what your change made unnecessary**.

### Imports
- Remove unused imports.
- Avoid wildcard imports (`from x import *`).
- Avoid import-time side effects.
- If imports exist only for re-export/public API:
  - make the intent explicit (e.g., `__all__`, or a comment), and ensure it matches repo conventions.

### Variables
- Remove unused variables.
- If a value is intentionally unused, name it `_` (or `_name`) to be explicit.
- Avoid “debug leftovers” (`tmp`, commented blocks, stray prints).

### Functions / classes
- Remove unused internal functions/classes you introduced or made obsolete.
- Do **not** delete exported/public API code unless the request explicitly requires it or you can prove it is unused across the repo.
- If removal could be risky, prefer deprecation patterns used in the repo (and document it).

### Files
- Don’t leave behind orphan modules after moving logic.
- Don’t create new utility modules for one-off helpers—extract only when reuse is real or imminent.

---

## 7) Large file & complexity guardrail
### File size rule
If you need to modify a file that is **> 500 lines**, treat that as a signal the module is doing too much.

Preferred actions (choose the least risky that still improves structure):
- Extract cohesive sections into smaller modules (e.g., `*_utils.py`, `*_types.py`, `*_client.py`).
- Split large functions into smaller single-purpose functions.
- Introduce clear boundaries (CLI vs orchestration vs integrations vs utilities).

Do **not** do a giant refactor. Keep it safe:
- Move only code you touch or the most cohesive block.
- Preserve public APIs and output schemas.
- Add/update tests to prove behavior is unchanged.

Hard requirement:
- Avoid adding new features into a >500-line file if a new focused module is the cleaner home.

### Complexity rule
- If a function is hard to read quickly, it’s too complex.
- Prefer early returns, clear naming, and small functions.
- Avoid deep nesting and over-engineering.

---

## 8) Error handling & robustness
- Fail early on invalid input and missing config.
- For network calls:
  - keep timeouts short and deterministic
  - propagate errors with useful context (include URL/service name, not secrets)
  - record/return error strings in output records where applicable
- Don’t swallow exceptions silently.
- Prefer explicit, actionable error messages over clever recovery.

### LLM/structured-output rule (if applicable)
- Request structured output (JSON/schema).
- Validate and normalize before use.
- Never write raw unchecked model output into durable artifacts.

---

## 9) Logging & debugging
- Prefer the repo’s logging approach over `print`.
- Log at boundaries and major decisions; avoid noisy logs in hot loops.
- Include identifiers that help debugging (request IDs, entity IDs).
- Never log secrets, tokens, credentials, or sensitive payloads.

---

## 10) Output/API stability (contracts)
Outputs are contracts.

- Do not rename/remove/repurpose output fields casually.
- If you must change a schema:
  - update docs and schema references
  - update downstream code/tests
  - call out the change explicitly in your summary
- Backward compatibility is preferred.

---

## 11) Testing expectations (practical, deterministic)
- Add or update tests whenever behavior changes.
- Minimum expectation: at least one test proving the behavior (plus an edge/failure case when feasible).
- Tests must be deterministic:
  - no real network calls in unit tests
  - mock time/randomness/external services
- If the repo lacks tests/harness:
  - add a minimal repro script OR provide explicit manual verification commands/steps.

---

## 12) Tooling & quality gates (run what exists)
Run the repo’s standard checks when possible:
- tests (unit/integration)
- formatter
- linter
- type checker (if used)

Do not introduce a new tool “because it’s better” unless explicitly requested.

Quality gate principle: **leave the repo cleaner than you found it** in the areas you touched
(e.g., removing newly-unused imports/vars, fixing obvious lint issues introduced by your change).

---

## 13) Security rules (non-negotiable)
- Never commit secrets/tokens/credentials.
- Never log secrets or sensitive data.
- Validate and sanitize inputs at boundaries.
- Avoid dangerous patterns (`eval`, unsafe YAML loading, shell injection, SQL injection).
- Do not weaken authentication/permissions/encryption for convenience.

---

## 14) Portability & “language-standard mindset” (C23-inspired, adapted for Python)
Write code that behaves predictably across environments.

- Prefer **defined behavior** over “it happens to work on my machine”.
- Avoid relying on implementation quirks (e.g., CPython-specific behavior) unless the repo explicitly targets that.
- Be explicit about:
  - encodings (`utf-8` unless otherwise required)
  - newline handling
  - timezone assumptions
  - filesystem paths (prefer `pathlib`)
- Resource management should be explicit:
  - use context managers (`with`) for files/locks/connections
- Be clear about limits:
  - max input size, timeouts, retries, memory growth
  - validate constraints at boundaries

Goal: portability, reliability, maintainability.

---

## 15) Self-review checklist (code review standards baked in)
Before you finalize, check:

### Design
- Does the change belong where it is implemented?
- Are responsibilities separated cleanly?
- Does it integrate with existing patterns?

### Functionality
- Does it do what the user asked (and nothing extra)?
- Are edge cases handled?
- Any concurrency hazards or race conditions introduced?

### Complexity
- Is any part harder than necessary to understand?
- Did you avoid over-engineering?

### Tests
- Are tests present and meaningful?
- Will they fail when the code is broken?

### Naming & readability
- Are names specific and clear?
- Is the code readable without comments?
- Comments (if any) explain **why**, not **what**.

### Style & consistency
- Conforms to repo style and conventions.
- No giant formatting-only diffs mixed into functional changes.

### Cleanup
- No unused imports, unused variables, dead functions introduced by the change.
- No leftover debug prints or commented-out blocks.

### Security & privacy
- No secrets in code/logs.
- Inputs validated and safe.

### Docs/contracts
- Output schemas stable, docs updated if changed.

---

## 16) Agent output requirements (what to report back)
When delivering changes, always include:
1) Summary of changes (bullets)
2) Files touched
3) How to run/test (exact commands)
4) Assumptions / trade-offs
5) Any follow-ups (optional, small)

If something couldn’t be verified locally, say so clearly and list what should be run.

---

## 17) Do-not-do list
- Do not refactor unrelated code.
- Do not invent APIs/data fields that don’t exist.
- Do not change schemas/contracts silently.
- Do not add telemetry/tracking without explicit permission.
- Do not add dependencies without strong justification.
- Do not perform heavy work at import time.
- Do not optimize prematurely.

---

## 18) Final checklist
- [ ] Minimal diff, correct behavior
- [ ] Boundaries respected (CLI/orchestration/integrations separated)
- [ ] Config centralized, constants not scattered
- [ ] No dead code or unused imports/vars
- [ ] Deterministic tests or clear manual verification
- [ ] Stable outputs / docs updated if required
- [ ] No secrets, safe input handling
