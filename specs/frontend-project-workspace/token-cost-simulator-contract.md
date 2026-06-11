# Token Cost Simulator Contract

## Purpose

Provide transparent estimates before provider calls and reconcile them with
provider-reported actual usage when available.

## Required Fields

- provider
- model
- pricing source
- pricing effective date
- input token estimate
- output token estimate
- actual input tokens
- actual output tokens
- estimated cost
- actual cost
- currency
- budget cap

## Rules

- Estimated costs must be labeled as estimates.
- Actual costs must be stored separately from estimates.
- Continuous generation must show a budget/cost warning before starting.
- Pricing tables are local and user-editable unless an approved online refresh
  is implemented later.

## Acceptance Criteria

- Known fixtures produce deterministic estimates.
- Cost warnings trigger before budget cap is exceeded.
- Reports show estimated versus actual usage separately.
