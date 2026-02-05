# Introduction

Many AI initiatives fail to transition from promising demonstrations to reliable products. A common failure mode is the absence of explicit requirements and measurable success criteria; model quality is discussed in isolation from user experience, cost, and operational constraints.

This text provides a product-driven, model-agnostic procedure for building AI products end-to-end. The workflow proceeds from specification of constraints, to model selection and validation, to fine-tuning decisions, to evaluation harness construction, and finally to an agent implementation suitable for production environments.

A single running example is used throughout: an **academic research assistant** that answers literature questions with verifiable citations, supported by retrieval-augmented generation (RAG), context engineering, knowledge-graph-based disambiguation and context expansion, targeted fine-tuning, and tool integrations for paper search and PDF parsing.

## How to use this book
- Begin with Part 1 to define constraints prior to model comparison.
- Interpret each chapter as a procedure that produces an artifact (e.g., a constraints sheet, a shortlist, a decision log).
- Apply the checklists to ensure that design choices remain linked to measurable outcomes.

## References
- Lewis, P., Perez, E., Piktus, A., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.
- Wu, L., et al. (2020). Literature Survey: Knowledge Graphs for Scholarly Data (representative background).
