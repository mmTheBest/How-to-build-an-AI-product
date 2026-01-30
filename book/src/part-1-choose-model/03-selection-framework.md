# Build a selection framework you can defend

## Opening

Shortlists reduce options, but you still need a decision. A framework makes your choice explicit, repeatable, and easy to explain to stakeholders.

This chapter gives you a simple scoring model you can adapt to any AI product.

## Core idea: score models against weighted constraints

### 1) Choose weights
Pick weights based on what matters most. For a sales email copilot, quality and latency often outrank cost, but cost still matters.

Example weights:
- Quality: 0.4
- Latency: 0.3
- Cost: 0.2
- Compliance: 0.1

### 2) Normalize your metrics
Convert metrics to 0–1 scales so they can be combined.

```text
normalized_latency = 1 - (latency / latency_budget)
normalized_cost = 1 - (cost / cost_budget)
```

### 3) Compute a weighted score
```text
score = 0.4*quality + 0.3*latency + 0.2*cost + 0.1*compliance
```

The exact math matters less than consistency and transparency.

## Example: decision matrix for the copilot

| Model | Quality | Latency | Cost | Compliance | Weighted score |
|---|---:|---:|---:|---:|---:|
| Hosted large | 0.85 | 0.60 | 0.30 | 0.70 | 0.64 |
| Hosted mid | 0.75 | 0.90 | 0.80 | 0.70 | 0.81 |
| OSS 7B | 0.60 | 0.40 | 0.90 | 0.90 | 0.62 |

The hosted mid model wins because it balances quality with speed and cost. The framework makes the trade‑off explicit instead of political.

## Checklist
- Define 3–5 decision criteria and weights
- Normalize metrics using your budgets
- Score each model using the same rubric
- Record the decision and the rationale in a short log

## Takeaway

A simple scoring framework turns model choice into an engineering decision, not a debate.
