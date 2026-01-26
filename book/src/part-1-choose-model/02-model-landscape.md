# Map the model landscape before you shortlist

## Opening

Once you have constraints, you need a map. “Best model” is meaningless without context. The landscape includes hosted APIs, open‑source models, and trade‑offs across quality, latency, cost, and operational complexity.

This chapter gives you a fast way to build a shortlist without committing to a stack too early.

## Core idea: models are a portfolio of trade‑offs

### Hosted API models
- **Pros:** fastest to ship, strong baseline quality, no infra setup.
- **Cons:** per‑request cost, data‑handling constraints, vendor lock‑in risk.

### Open‑source models
- **Pros:** full control, lower marginal cost at scale, on‑prem options.
- **Cons:** infra overhead, tuning/serving complexity, quality variance.

### Model size tiers
- **Small models:** lower cost, faster latency, weaker reasoning.
- **Large models:** higher quality, higher cost, slower responses.

## Quick sweep: build a baseline table

Start with a small dataset (50–100 examples). Score a few candidate models on your core constraints.

```python
# Pseudocode: quick sweep scoring
candidates = ["hosted-large", "hosted-mid", "oss-7b"]
for model in candidates:
    results = evaluate(model, dataset)
    print(model, results["approval_rate"], results["p95_latency"], results["cost_per_email"])
```

You are not picking a winner yet. You are eliminating models that obviously fail.

## Example: sales email copilot shortlist

Illustrative baseline on 100 labeled emails:

| Model | Approval rate | p95 latency | Cost/email | Notes |
|---|---:|---:|---:|---|
| Hosted large | 78% | 1.9s | $0.07 | quality strong, cost too high |
| Hosted mid | 70% | 1.1s | $0.03 | meets constraints |
| OSS 7B | 62% | 2.8s | $0.01 | needs GPU to hit latency |

The sweep narrows the field. In this example, “hosted mid” is the only option that hits all constraints without extra infrastructure.

## Checklist
- List 3–5 candidate models across hosted + open‑source
- Run a small baseline eval on real examples
- Compare against your hard constraints first
- Keep only the models that pass the first filter

## Takeaway

A quick sweep turns model choice from opinion into evidence. Use it to build a shortlist you can test deeply.
