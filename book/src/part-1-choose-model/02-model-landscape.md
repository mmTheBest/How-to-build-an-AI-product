# Map the model landscape prior to shortlisting

## Introduction

Given a constraints specification, the next step is to construct a shortlist of candidate models. [Beyer et al., 2016]
The objective is not to identify a universally “best” model, but to eliminate candidates that are incompatible with latency, cost, reliability, and evidence requirements. [Jain, 1991]

This chapter provides a compact taxonomy of the model landscape and a baseline procedure for shortlisting candidates.

## 2.1 Primary axes of the model landscape

### 2.1.1 Hosted API models versus self-hosted (open-weight) models

Hosted API models are accessed via commercial providers.
They typically reduce integration time and eliminate model serving overhead, at the cost of per-request pricing, dependency on external reliability, and constraints on data handling. [Beyer et al., 2016]

Self-hosted (open-weight) models are operated under organizational control.
They can offer stronger control over data governance and potentially lower marginal cost at scale, but require engineering effort for serving, scaling, observability, and incident response. [Beyer et al., 2016]

### 2.1.2 Model size tiers and operational consequences

Model size is a coarse proxy for capability, but it is also a proxy for serving cost, latency, and operational complexity.
Smaller models are typically faster and cheaper to run, but may be less reliable in instruction following and structured tool use under distribution shift.
Larger models often improve robustness and tool-use reliability, but can violate cost and tail-latency constraints unless mitigated by system design.

For AI agent products, tool-use reliability can become a binding constraint: a single failed tool call can dominate end-to-end failure rates even when free-form answer quality is high.

## 2.2 Baseline sweep procedure (shortlist construction)

A baseline sweep is used to filter clearly incompatible candidates prior to deeper integration.

1. Construct a small evaluation set (e.g., 50–100 prompts) representative of expected product usage.
2. Define metrics aligned with constraints (e.g., evidence quality, tool-call success rate, p95/p99 latency, and marginal cost).
3. Evaluate a small portfolio spanning hosted and self-hosted candidates.
4. Eliminate candidates that violate hard constraints; retain artifacts for later regression testing.

```python
# Pseudocode: baseline sweep
candidates = ["hosted-large", "hosted-mid", "oss-7b"]
for model in candidates:
    r = evaluate(model, dataset)
    print(
        model,
        r["claim_support_rate"],
        r["citation_precision"],
        r["tool_call_success"],
        r["p95_latency"],
        r["cost_per_answer"],
    )
```

## 2.3 Chapter remainder (outline)

The remainder of this chapter is retained as an outline and will be expanded in later work.

- Add a worked RA example of a shortlist table.
- Add a discussion of common shortlisting failure modes (e.g., benchmark chasing, ignoring tail latency).
- Add a references-backed discussion of tool-use reliability as a system property.

## References
- Beyer, B., Jones, C., Petoff, J., & Murphy, N. R. (Eds.). (2016). *Site Reliability Engineering: How Google Runs Production Systems*. O’Reilly Media.
- Jain, R. (1991). *The Art of Computer Systems Performance Analysis: Techniques for Experimental Design, Measurement, Simulation, and Modeling*. Wiley.
