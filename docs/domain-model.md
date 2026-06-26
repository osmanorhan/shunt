---
nav_exclude: true
---

# Domain Model

The first implementation should revolve around six objects.

## 1. TaskRun

`TaskRun` is the live task container.
It tracks:

- current phase
- current understanding artifact
- active recipe run
- linked frontier cases

This is the runtime anchor.

## 2. UnderstandingArtifact

`UnderstandingArtifact` is the teachback object shown before execution.
It captures:

- original request
- interpreted goal
- success criteria
- constraints and target scope
- evidence
- assumptions
- ambiguities
- selected recipe
- risks
- approval state

If this object is weak, the rest of the system will drift.

## 3. RecipeRun

`RecipeRun` is the staged execution record.
It does not own task understanding.
It records how a selected recipe was executed through:

- inspect
- propose
- verify
- apply
- validate

## 4. UncertaintyEvent

`UncertaintyEvent` is the smallest unit of instability.
It records things like:

- missing evidence
- low confidence
- verifier disagreement
- retry exhaustion
- user correction

This should exist early, not as a later optimization.

## 5. FrontierCase

`FrontierCase` is created when uncertainty becomes important enough to preserve.
It links:

- the task
- the current understanding state
- the recipe run, if any
- the instability reason
- the relevant uncertainty events

This is the bridge from runtime to improvement.

## 6. CorrectionPackage and AdaptationPackage

`CorrectionPackage` stores a structured repair for a frontier case.
`AdaptationPackage` stores something validated enough to promote back into runtime behavior.

They are separate because repair and promotion are not the same step.

## Core Invariants

- nothing executes before an `UnderstandingArtifact` exists
- user correction updates the artifact, not just chat history
- execution is staged and verifier-visible
- instability is recorded as structured data
- repairs are stored in the same shape the runtime already uses
- promoted improvements are explicit and traceable