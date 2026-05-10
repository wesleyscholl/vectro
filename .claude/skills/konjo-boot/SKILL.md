---
name: konjo-boot
description: Boot a Konjo session for lopi. Produces a Session Brief, runs Discovery, identifies the next sprint. Use at the start of any work session or when invoked with /konjo.
user-invocable: true
---
# Konjo Session Boot — lopi

## Step 1 — Read
Read in order: CLAUDE.md, README.md, CHANGELOG.md, PLAN.md, MASTER_PLAN.md, docs/ (if it exists).
Do not skip. Do not assume contents.

## Step 2 — Session Brief
```
REPO         lopi — high-performance Rust agent orchestrator for Claude Code
LAST SHIPPED [most recent meaningful change from CHANGELOG.md]
OPEN WORK    [stated next steps from PLAN.md]
BLOCKERS     [failing tests, broken modules, open issues]
HEALTH       [Green / Yellow / Red — one line]
```
Unknown is stated as unknown. Fabricated state is a lie to the next session.

## Step 3 — Discovery (कोहजो)
Before executing any sprint, ask:
- What has shipped in the agent orchestration / Claude Code ecosystem since this repo last moved?
- Are there new papers, repos, or techniques relevant to lopi's core problems (agent scheduling, memory, self-improvement)?
- What would a researcher building in this space know today that this repo doesn't reflect?

Search: arXiv (agent scheduling, LLM orchestration, self-improving systems), GitHub (new Claude Code tooling, agent frameworks), HuggingFace (relevant models/benchmarks).

## Step 4 — Identify Work
If PLAN.md exists: load it, validate against codebase, flag drift.
If drift found:
```
PLAN DRIFT
  ✗ [item] appears completed — not marked done
  ✗ [item] references [module] that no longer exists
  CORRECTED NEXT STEP: [what actually needs to happen]
```
If no plan: run Discovery Protocol → propose a sprint:
```
PROPOSED SPRINT  [N — Name]
MOTIVATION       [real problem, real user, real value]
RESEARCH         [papers, repos, techniques informing this sprint]
DELIVERABLES     [concrete, shippable, verifiable things]
SUCCESS CRITERIA [tests, benchmarks, features working end-to-end]
SCOPE / RISKS    [Small / Medium / Large — what could block this]
```
Small/Medium: propose and proceed. Large or irreversible: propose and confirm.

## Invocation Keywords
This skill activates on any of:
- `konjo`
- `konjo lopi`
- `lopi konjo`
- `read KONJO_PROMPT.md and begin`
