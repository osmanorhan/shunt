---
nav_exclude: true
---

# Iterations

This project should treat the architecture draft as guidance, not doctrine.

Section 18 is the operating thesis to validate:

- understanding before execution
- staged execution instead of a free agent loop
- uncertainty-gated control
- frontier capture during live runs
- correction in runtime-native formats
- promotion only after validation

Each iteration should stay onion-small, then be tested against a local `llama-server`.
If the artifacts or control loop feel wrong in practice, refactor early.

## Iteration 1: Onion Core

Goal: prove the task loop and artifacts, not broad task completion.

Build:

- core domain model
- local persistence
- clarify, understand, agree flow
- one local provider path
- one narrow recipe shape
- uncertainty ledger
- frontier capture

Validate:

- the model produces a usable understanding artifact
- user revisions improve the artifact rather than restart the task
- uncertainty signals explain weak or unstable steps
- failed runs leave behind useful frontier cases

Exit:

- 5-10 real tasks run through the loop
- at least one task completes after approval
- failures are understandable from stored artifacts alone

## Iteration 2: Verified Recipe Execution

Goal: add bounded execution under verifier control.

Build:

- recipe selection
- staged execution: inspect, propose, verify, apply, validate
- minimal verifiers for shape, patchability, and command or test outcomes

Validate:

- a narrow task class succeeds more often than a free-form loop would
- verifier failures block bad execution early
- stage records are easy to inspect

Exit:

- one narrow repository task class can complete end to end
- unstable runs still emit frontier cases

## Iteration 3: Correction Loop

Goal: turn failure into reusable repairs.

Build:

- replay for failed runs
- maintainer correction flow
- correction packages tied to frontier cases
- revalidation after repair

Validate:

- a failed case can be reopened and repaired without leaving the task model
- repairs are stored as structured runtime objects, not notes

Exit:

- corrected cases can be re-run and compared to the original failure

## Iteration 4: Promotion

Goal: make validated repairs change future behavior.

Build:

- proposal, validation, promotion states
- recipe updates from corrected cases
- node or threshold updates from corrected cases
- optional adaptation exports

Validate:

- prior repairs improve future runs without manual one-off edits
- promotions are explicit and traceable

Exit:

- the runtime can consume promoted updates safely

## Working Rule

After each iteration:

1. run real tasks against local `llama-server`
2. inspect artifacts and frontier cases
3. decide what to simplify, tighten, or refactor
4. only then expand scope