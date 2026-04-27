# How to Build an AI Product From Scratch

This repository is a writing-first playbook for turning impressive AI demos into dependable products. It uses one concrete example throughout: **Arxie**, an academic research assistant that retrieves real papers, reads full text, and returns citation-grounded outputs.

The goal is practical rather than promotional. Each chapter is meant to make one class of product decisions explicit: model selection, context engineering, evaluation, architecture, operations, and the point at which prompting stops being enough.

## Start Here

- If you want the most complete material today, start with [Chapter 1](chapters/01-model-selection.md) and [Chapter 2](chapters/02-context-engineering.md).
- If you want the roadmap for what is coming next, read the outline chapters and the [Arxie artifact map](appendices/arxie-artifact-map.md).

## Canonical Example

The manuscript uses **Arxie** as the sole running example.

Arxie is an academic research assistant that:

- searches live academic sources
- reads full papers rather than only abstracts
- formats inline citations and references
- evaluates outputs against explicit quality metrics
- exposes concrete engineering artifacts for prompt design, retrieval, evaluation, and release gates

This constraint is intentional. One stable example is more useful than several shallow ones.

## Current Manuscript Status

| Chapter | Focus | Status |
|---|---|---|
| 1 | Model selection | Draft |
| 2 | Context engineering and prompt design | Draft |
| 3 | Evaluation and release gates | Outline |
| 4 | Agent architecture | Outline |
| 5 | Shipping and operating AI products | Outline |
| 6 | When prompting stops working | Outline |

## Reading Order

1. [Chapter 1: Model Selection](chapters/01-model-selection.md)
2. [Chapter 2: Context Engineering and Prompt Design](chapters/02-context-engineering.md)
3. [Chapter 3: Evaluation and Release Gates](chapters/03-evaluation-and-release-gates.md)
4. [Chapter 4: Agent Architecture](chapters/04-agent-architecture.md)
5. [Chapter 5: Shipping and Operating AI Products](chapters/05-shipping-and-operations.md)
6. [Chapter 6: When Prompting Stops Working](chapters/06-when-prompting-stops-working.md)

## What Changed Today

- The manuscript was split into chapter pages so readers no longer have to scroll through one monolithic file.
- The remaining chapter sequence was reordered to match Arxie's real development logic: prompting is followed by evaluation, then architecture, then operations.
- Chapter 2 was extended with a new section on the prompt engineering ceiling.

## Working Style

This repo is updated in small daily increments. The public chapters are intended to stay readable even when some later chapters are still outline-only.
