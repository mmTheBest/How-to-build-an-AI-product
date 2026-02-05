# Define a defensible selection framework

## Introduction

Shortlisting reduces the model search space but does not determine a final choice.
A defensible selection framework makes model choice explicit, repeatable, and auditable, and reduces the risk that selection is driven by subjective demonstrations.

This chapter specifies a minimal decision procedure that combines (i) a hard-constraint filter and (ii) a transparent scoring rule over secondary criteria.

## 3.1 Hard constraints as decision gates

Hard constraints define feasibility.
Any candidate violating at least one hard constraint is excluded from further consideration, even if it scores highly on other criteria.
This “gate first” structure prevents averaging from masking unacceptable failures.

## 3.2 Weighted scoring among feasible candidates

Among feasible candidates, a weighted scoring model can be used to encode explicit product priorities.
The purpose of the scoring model is not mathematical sophistication but transparency and reproducibility.

### 3.2.1 Criteria and weights

Criteria should be derived from the constraint specification.
For evidence-grounded agent products (including an academic research assistant), evidence quality and tool-use reliability are typically first-order.

Example criteria and weights:
- Evidence quality: 0.35
- Tool-use reliability: 0.20
- Latency: 0.20
- Cost: 0.15
- Disambiguation behavior: 0.10

### 3.2.2 Metric normalization

Metrics should be normalized to a comparable 0–1 scale with respect to budgets.
For budgeted quantities such as latency and cost, a simple normalization is:

```text
normalized_latency = max(0, 1 - (latency / latency_budget))
normalized_cost    = max(0, 1 - (cost / cost_budget))
```

Evidence-related metrics (e.g., claim-support rate, citation precision) may already lie in [0, 1] under a defined rubric.

### 3.2.3 Score computation

```text
score = Σ_i w_i * metric_i
```

The scoring stage should not be used to compensate for violations of hard constraints.

## 3.3 Decision log and re-evaluation

A selection decision should be recorded in a short decision log that includes:

- the constraint specification (hard/soft)
- candidate set
- evaluation artifacts (inputs, outputs, measurements)
- the scoring rule and weights
- the selected model and rationale

The decision should be revisited when constraints change (e.g., a new cost ceiling) or when model/provider characteristics change.

## 3.4 Chapter remainder (outline)

The remainder of this chapter is retained as an outline and will be expanded in later work.

- Add a worked decision matrix example for the RA.
- Add guidance for sensitivity analysis (how decisions change with weight changes).

## References (outline)
- Add books/papers on multi-criteria decision analysis and evaluation methodology.
