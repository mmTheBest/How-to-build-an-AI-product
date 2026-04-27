# Chapter 3: Evaluation and Release Gates

> Status: outline anchored to Arxie's implemented evaluation stack.

This chapter follows context engineering because prompt work is only meaningful when it is tied to repeated measurement. Arxie already contains a baseline evaluation report, an eval harness, and proposal release-gate machinery, so evaluation is the next concrete layer of the example rather than a late-stage appendix.

**Primary Arxie artifacts**

- `tests/eval/harness.py`
- `docs/eval-baseline.md`
- `src/ra/eval/release_gate.py`
- `tests/eval/test_eval_harness.py`
- `tests/eval/test_proposal_release_gate.py`

## 3.1 Evaluation as product infrastructure

This section will argue that evaluation is not a report produced at launch time but a permanent subsystem that converts product requirements into executable checks.

## 3.2 Defining "done" with measurable gates

This section will show how hard constraints, soft constraints, and release criteria become the practical definition of shippable quality.

## 3.3 Dataset construction and versioning

This section will cover representative query selection, stratification, minimum viable dataset size, and why frozen versions are required for comparison across prompt and model changes.

## 3.4 Metric design

This section will separate metrics by failure mode: citation precision, claim support, tool success, latency, and format reliability.

## 3.5 Automated and human evaluation

This section will explain which judgments can be automated safely, which require human review, and how to combine both without losing reproducibility.

## 3.6 Regression testing and release gates

This section will use Arxie's proposal release-gate evaluator to show how evaluation moves from offline reporting into deployment policy.

## 3.7 Drift, maintenance, and re-evaluation

This section will cover dataset refresh, production-failure capture, provider drift, and the conditions that trigger re-running earlier decisions.
