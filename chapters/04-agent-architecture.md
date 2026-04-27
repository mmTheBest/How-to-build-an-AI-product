# Chapter 4: Agent Architecture

> Status: outline anchored to Arxie's implemented agent, tool, retrieval, and proposal layers.

This chapter turns from evaluation to the system that is being evaluated. In Arxie, the interesting engineering work does not live in the model call alone; it lives in the loop that retrieves papers, reads full text, formats citations, and decides when the agent has enough evidence to answer.

**Primary Arxie artifacts**

- `src/ra/agents/research_agent.py`
- `src/ra/tools/retrieval_tools.py`
- `src/ra/retrieval/`
- `src/ra/parsing/pdf_parser.py`
- `src/ra/proposal/`

## 4.1 The agent loop

This section will explain the observe-think-act loop, why tool-based loops outperform single-pass prompting for research tasks, and where that flexibility becomes expensive.

## 4.2 Retrieval system design

This section will cover source selection, deduplication, caching, freshness, and the tradeoff between broad recall and controllable latency.

## 4.3 Tool design and schema boundaries

This section will use Arxie's retrieval tools to show how tool granularity, argument descriptions, and output shapes affect reliability.

## 4.4 Full-text access and evidence extraction

This section will explain why full-text analysis changes product capability materially and why it belongs to architecture rather than prompting.

## 4.5 State, memory, and multi-turn work

This section will examine chat sessions, conversation compression, and the boundary between transient context and durable research state.

## 4.6 Post-processing and confidence layers

This section will cover citation formatting, claim extraction, confidence scoring, and the role of deterministic post-processing in a probabilistic system.

## 4.7 Failure handling and graceful degradation

This section will classify tool failures, retrieval failures, and loop failures, then connect them to fallback policy and user-visible behavior.
