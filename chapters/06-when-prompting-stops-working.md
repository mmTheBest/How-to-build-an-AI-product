# Chapter 6: When Prompting Stops Working

> Status: outline reframed around the escalation path after prompt, system, and architecture improvements plateau.

This chapter keeps fine-tuning in the book, but moves it to the correct place in the decision sequence. In Arxie, the stronger immediate lessons come from prompting, evaluation, retrieval, and operations. Model customization becomes relevant only after those layers are stable enough that remaining failures can be attributed to the model rather than to missing system design.

**Primary Arxie artifacts**

- `docs/eval-baseline.md`
- `docs/PRD.md`
- `DEVLOG.md`
- `src/ra/agents/research_agent.py`

## 6.1 The escalation sequence

This section will define the order of interventions: prompt, tool description, post-processing, retrieval architecture, operating policy, and only then model customization.

## 6.2 Which failures are model failures

This section will separate reasoning gaps, calibration limits, and domain adaptation problems from failures that were misdiagnosed but actually belong elsewhere.

## 6.3 When fine-tuning is justified

This section will provide the decision rule for entering a fine-tuning program: stable eval harness, clear residual failure pattern, economic reason to customize, and a credible data path.

## 6.4 Data requirements and curation

This section will cover how data quality, label design, and sampling strategy dominate fine-tuning outcomes.

## 6.5 Strategy choices

This section will compare full fine-tuning, LoRA, adapters, and smaller customization programs from the perspective of product cost and maintenance burden.

## 6.6 The maintenance tax

This section will explain why every customized model adds versioning, regression, and operational overhead that must be justified explicitly.

## 6.7 Arxie as the cautionary example

This section will argue that Arxie is valuable precisely because it shows how far a product can go before fine-tuning is the right next move.
