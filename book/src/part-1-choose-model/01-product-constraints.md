# Define product constraints before you pick a model

## Opening

You can build a sales email copilot in a weekend with any modern model. The hard part is making it useful on Monday. If you don’t define constraints up front, you can’t tell whether a model is “good enough,” and you’ll spend weeks debating demos instead of shipping.

This chapter turns fuzzy goals into measurable constraints. By the end, you’ll have a one‑page spec you can use to compare models fairly.

## Core idea: constraints come first

Start with the product, not the model. The model is one component inside a system, and its value depends on your constraints.

### 1) User experience constraints
- **Latency:** how fast the draft must appear (p95 or p99).
- **Interaction pattern:** one-shot generation vs. iterative suggestions.
- **Tone control:** brand-safe language and consistent style.

### 2) Business constraints
- **Cost per output:** how much each email can cost and still be viable.
- **Value per output:** how you measure impact (reply rate, time saved).
- **Scale:** expected volume per day or per user.

### 3) Technical constraints
- **Context length:** how much CRM data you need per email.
- **Throughput:** requests per second during peak usage.
- **Reliability:** acceptable failure rate and fallback behavior.

### 4) Compliance constraints
- **PII handling:** what can leave your environment.
- **Data retention:** whether prompts/outputs can be stored.
- **Auditability:** ability to explain or reproduce outputs.

## Turn constraints into metrics

Write each constraint as a measurable target. A model choice becomes easy when the targets are clear.

| Constraint | Metric | Example target |
|---|---|---|
| Speed | p95 latency | ≤ 2.0s |
| Cost | $ per email | ≤ $0.03 |
| Quality | approval rate | ≥ 70% |
| Safety | policy violations | 0 per 1,000 |

## Example: sales email copilot constraints

The copilot drafts short, personalized emails from CRM notes. Sales reps edit before sending. The product constraints could look like this:

- **Latency:** p95 ≤ 2.0s for a single draft
- **Cost:** ≤ $0.03 per email at 500–700 tokens
- **Quality:** ≥ 70% “send with minor edits” in a 100‑sample review
- **Safety:** 0 policy violations in a weekly spot check

These numbers create a clean decision boundary. A model that misses two constraints is not a fit, even if it looks impressive in a demo.

## Checklist
- Write a one‑sentence product goal (what changes for the user?)
- Pick 3–5 measurable constraints tied to that goal
- Set concrete targets (p95 latency, cost per output, approval rate)
- Decide which constraints are hard stops vs. flexible

## Takeaway

A model is only “good” relative to your constraints. Write them down first, then compare options.
