# How to Build an AI Product From Scratch

Many AI initiatives fail to transition from promising demonstrations to reliable products.
A common cause is the absence of explicit requirements and measurable success criteria; model selection is then driven by subjective impressions rather than product constraints.

This repository is a writing-first playbook for designing AI products end-to-end.
A single running example is used only to instantiate definitions and measurement choices: an **academic research assistant** (RA) that answers literature questions with verifiable citations, supported by retrieval-augmented generation (RAG), context engineering, knowledge-graph-based disambiguation and context expansion, targeted fine-tuning, and tool integrations for paper search and PDF parsing.

---

## Chapter 1: Model Selection

Model selection is only meaningful relative to the product constraints under which the system must operate. [Beyer et al., 2016]
In the absence of explicit constraints, model choice tends to be driven by qualitative demonstrations and benchmark scores that are weakly coupled to end-to-end product performance. [Jain, 1991]

The objective of this chapter is to provide a repeatable procedure for translating informal product requirements into measurable targets that can be used to (i) eliminate infeasible design options, (ii) compare candidate models fairly, and (iii) define evaluation gates for subsequent iteration. [Beyer et al., 2016]
The running example is an academic research assistant (RA), which is used only to instantiate definitions and measurement choices. [Lewis et al., 2020]

### 1.1 Model selection as a constraint satisfaction problem

Model selection can be formalized as a constrained decision problem. [Keeney & Raiffa, 1993]
The relevant object of optimization is end-to-end system behavior, because the model's utility depends on retrieval quality, tool latency, prompt structure, and downstream validation. [Beyer et al., 2016]

Let \(M\) denote a set of candidate models or model configurations and let \(C\) denote a set of constraints derived from product requirements. [Keeney & Raiffa, 1993]
Constraints should be partitioned into **hard constraints** \(H\subset C\), which function as decision gates, and **soft constraints** \(S\subset C\), which are optimized within the feasible region. [Beyer et al., 2016]

The model selection procedure is then:

1. **Feasibility filtering:** eliminate any candidate \(m\in M\) that violates at least one hard constraint in \(H\). [Beyer et al., 2016]
2. **Optimization among feasible candidates:** select the candidate that optimizes an explicit objective over \(S\) (e.g., a weighted score), conditional on satisfying \(H\). [Keeney & Raiffa, 1993]

This separation is important because averaging criteria can mask violations of requirements that are non-negotiable for the product. [Beyer et al., 2016]

Consider an academic research assistant designed to answer questions about literature. A researcher might ask:

> "What are the main critiques of BERT's tokenization approach in the recent NLP literature?"

The RA must:
1. **Understand the query scope** — identify that "BERT tokenization" refers to WordPiece tokenization, "recent literature" likely means papers from the last 3-5 years, and "critiques" implies negative findings or limitations.
2. **Retrieve relevant sources** — search academic databases (e.g., Semantic Scholar, arXiv) for papers discussing BERT tokenization limitations.
3. **Extract claims** — identify specific critiques from paper abstracts or full text (e.g., "WordPiece fails on morphologically rich languages" or "subword tokenization loses semantic compositionality").
4. **Cite accurately** — provide precise citations (author, year, title) for each claim.
5. **Respond with latency** — deliver the answer within 3-5 seconds (user expectation for interactive research).

Each of these requirements translates into measurable constraints:

| Requirement | Constraint Type | Metric | Threshold |
|-------------|----------------|--------|-----------|
| Query understanding | Soft | Disambiguation success rate | ≥0.85 |
| Source retrieval | Hard | Retrieval recall@10 | ≥0.90 |
| Claim extraction | Soft | Claim-support rate (manual eval) | ≥0.80 |
| Citation accuracy | Hard | Citation precision | ≥0.85 |
| Response latency | Hard | p95 latency | ≤5.0s |
| Cost per query | Hard | Cost (API + retrieval) | ≤$0.15 |

**Hard vs. Soft Constraints:**
- **Hard:** Citation precision <0.85 is unacceptable (fabricated citations undermine trust and scholarly value). Latency >5s violates user experience expectations. Cost >$0.15 makes the product economically unviable at scale.
- **Soft:** Higher claim-support rate is better (0.90 > 0.80), but we can tolerate some ambiguity if other factors compensate. Disambiguation success can be improved with system design (e.g., asking clarifying questions).

### 1.2 Translating requirements into measurable targets

A requirement is not operational until it is expressed as an evaluable claim with a measurement protocol. [Jain, 1991]
For example, the statement "answers should be well supported" can be operationalized as a citation precision or claim-support metric computed on a fixed evaluation set under a defined rubric. [Rajpurkar et al., 2018]

Operationalization should specify:

- the **metric** (what is measured), [Jain, 1991]
- the **threshold** (what passes), [Beyer et al., 2016]
- the **population and conditions** (which requests, which workload), [Jain, 1991]
- the **measurement procedure** (instrumentation and estimator). [Jain, 1991]

When a target depends on human judgment (e.g., whether a claim is supported by evidence), the annotation procedure must be specified and versioned so that comparisons across model updates are interpretable. [Rajpurkar et al., 2018]

**Operationalizing "Citation Precision".**
Citation precision is the fraction of cited sources that actually support the claim they are attributed to.

*Measurement Protocol:*
1. **Test set construction:** Create 100 questions about known papers (ground truth available).
2. **Model response collection:** For each question, collect the RA's answer with citations.
3. **Claim-citation pairing:** Parse the response into (claim, citation) pairs. Example:
   ```
   Claim: "WordPiece tokenization fails on morphologically rich languages"
   Citation: Schuster & Nakajima (2012), "Japanese and Korean Voice Search"
   ```
4. **Verification:** For each pair, a human annotator checks:
   - Does the cited paper exist? (existence check)
   - Does the cited paper discuss the claim? (relevance check)
   - Does the cited paper support the claim as stated? (correctness check)
5. **Scoring:** Citation precision = (correct pairs) / (total pairs).

*Versioning:* Changes to the test set or annotation rubric must be versioned (e.g., "eval-v1.0") to ensure reproducibility across model updates.

*Why This Matters for Model Selection:*
A model that scores 0.75 on citation precision (25% fabricated or irrelevant citations) violates the hard constraint and is eliminated, regardless of how eloquent its prose. A model that scores 0.88 meets the threshold and proceeds to soft optimization.

### 1.3 Mapping the model landscape

Given a constraints specification, the next step is to construct a shortlist of candidate models. [Beyer et al., 2016]
The objective is not to identify a universally "best" model, but to eliminate candidates that are incompatible with latency, cost, reliability, and evidence requirements. [Jain, 1991]

#### 1.3.1 Hosted API models versus self-hosted (open-weight) models

Hosted API models are accessed via commercial providers.
They typically reduce integration time and eliminate model serving overhead, at the cost of per-request pricing, dependency on external reliability, and constraints on data handling. [Beyer et al., 2016]

Self-hosted (open-weight) models are operated under organizational control.
They can offer stronger control over data governance and potentially lower marginal cost at scale, but require engineering effort for serving, scaling, observability, and incident response. [Beyer et al., 2016]

#### 1.3.2 Model size tiers and operational consequences

Model size is a coarse proxy for capability, but it is also a proxy for serving cost, latency, and operational complexity.
Smaller models are typically faster and cheaper to run, but may be less reliable in instruction following and structured tool use under distribution shift.
Larger models often improve robustness and tool-use reliability, but can violate cost and tail-latency constraints unless mitigated by system design.

For AI agent products, tool-use reliability can become a binding constraint: a single failed tool call can dominate end-to-end failure rates even when free-form answer quality is high.

#### 1.3.3 Current model landscape (2026)

In the example of the RA agent, the landscape must be filtered by **tool-use reliability** (ability to correctly invoke paper search and PDF parsing tools) and **long-context handling** (processing full-text papers, which can be 8-12k tokens).

**Hosted API Models**

**GPT-5 Series (OpenAI):**
The GPT-5 family represents OpenAI's current flagship models, replacing the deprecated GPT-4 series.
- *GPT-5.2:* Optimized for coding and agentic tasks. Input pricing at $1.75/1M tokens, output at $14/1M tokens. Supports cached input at $0.175/1M tokens for repeated prompts.
- *GPT-5.2 Pro:* The highest-capability variant for complex reasoning. Significantly more expensive ($21/1M input, $168/1M output) but offers superior precision.
- *GPT-5 mini:* A faster, cheaper option ($0.25/1M input, $2/1M output) suitable for well-defined tasks where maximum capability is not required.
- *RA fit:* GPT-5.2 is the natural choice for agentic workflows requiring tool use. At ~3k tokens per query, cost is approximately $0.05–0.10 per answer (within budget). Strong function calling reliability.

**Claude 4 Series (Anthropic):**
The Claude 4 family has replaced Claude 2 and 3, with models optimized for different use cases.
- *Opus 4.6:* The most intelligent model, optimized for building agents and coding. Input at $5/1M tokens (≤200K context), output at $25/1M tokens. Supports prompt caching for cost reduction.
- *Sonnet 4.5:* Optimal balance of intelligence, cost, and speed. Input at $3/1M tokens, output at $15/1M tokens.
- *Haiku 4.5:* Fastest and most cost-efficient. Input at $1/1M tokens, output at $5/1M tokens.
- All models support 200K+ token context with tiered pricing for longer contexts.
- *RA fit:* Opus 4.6 excels at agent construction and reasoning. Sonnet 4.5 offers a strong balance for production use. The extended context window (200K+) allows processing multiple papers in a single call.

**Gemini 3/2.5 (Google):**
Google's Gemini models are natively multimodal, supporting text, image, video, and audio inputs.
- *Gemini 3 Pro Preview:* Input at $2/1M tokens, output at $12/1M tokens. Supports image output.
- *Gemini 3 Flash Preview:* Faster variant at $0.5/1M input, $3/1M output.
- *Gemini 2.5 Pro:* Input at $1.25/1M tokens, output at $10/1M tokens. Includes computer-use capabilities.
- *Gemini 2.5 Flash:* Cost-efficient at $0.30/1M input, $2.50/1M output.
- Grounding with Google Search available (5,000–10,000 free queries/day depending on model).
- *RA fit:* Strong candidate if multimodal processing is needed (interpreting figures, diagrams, or tables in papers). Web grounding can supplement paper search.

**Open-Weight Models**

**Llama 4 (Meta):**
The Llama 4 collection represents Meta's current flagship, featuring mixture-of-experts (MoE) architecture for efficient scaling.
- *Llama 4 Scout:* 17B parameters with 16 experts. Efficient for most tasks.
- *Llama 4 Maverick:* 17B parameters with 128 experts. Higher capability through more specialized routing.
- Natively multimodal (text and image understanding).
- Open weights under permissive license for commercial use.
- *RA fit:* Strong option for self-hosting if data privacy or cost-at-scale is critical. MoE architecture provides good capability/efficiency tradeoff. Requires infrastructure for serving (GPU cluster with vLLM or similar).

**Llama 3.3 (Meta):**
- 70B parameter text-only model, instruction-tuned.
- Simpler to deploy than Llama 4 (no MoE routing).
- Strong performance on knowledge tasks, competitive with hosted models.
- *RA fit:* Viable for teams with existing LLaMA infrastructure who want a stable, well-understood model.

**Mistral Small 3.2 / Magistral (Mistral AI):**
Mistral has evolved from Mixtral 8x7B (now deprecated) to the Magistral and Mistral Small series.
- *Magistral Medium/Small:* Current flagship models optimized for general tasks.
- *Mistral Small 3.2:* Efficient model for production deployment.
- *Devstral Small:* Specialized for coding tasks.
- Available via API and as open weights.
- *RA fit:* Mistral Small 3.2 offers a good balance of capability and efficiency for self-hosting. Less proven for academic QA than larger models.

**Cohere (Enterprise):**
Cohere has transitioned to enterprise-only pricing with their North platform (AI workspace) and Compass (intelligent search).
- No longer offers public API pricing for individual model access.
- Command R+ and similar models available through enterprise agreements.
- *RA fit:* Suitable for enterprise deployments with negotiated pricing. Not recommended for prototyping or small-scale projects.

**RA Landscape Decision Matrix (2026):**

| Model | Context | Tool Use | Est. Cost/Query | Deployment | RA Suitability |
|-------|---------|----------|-----------------|------------|----------------|
| GPT-5.2 | 128K+ | Excellent | $0.05-0.10 | API only | Strong, within budget |
| Opus 4.6 | 200K+ | Excellent | $0.08-0.15 | API only | Best for complex reasoning |
| Sonnet 4.5 | 200K+ | Very Good | $0.04-0.08 | API only | Balanced choice |
| Gemini 2.5 Pro | 200K+ | Good | $0.03-0.06 | API only | Multimodal strength |
| Llama 4 Maverick | Large | Good | $0.01-0.03* | Self-host | Privacy/cost at scale |
| Mistral Small 3.2 | 32K+ | Fair | $0.01-0.02* | Self-host/API | Efficient option |

*Self-hosted costs are infrastructure-dependent (GPU hours, not per-token).

**Shortlist for RA Baseline Sweep:**
1. **GPT-5.2** (hosted, agentic benchmark)
2. **Sonnet 4.5** (hosted, cost-performance balance)
3. **Llama 4 Maverick** (open, self-hosted option)

### 1.4 Baseline sweep procedure

A baseline sweep is used to filter clearly incompatible candidates prior to deeper integration.

1. Construct a small evaluation set (e.g., 50–100 prompts) representative of expected product usage.
2. Define metrics aligned with constraints (e.g., evidence quality, tool-call success rate, p95/p99 latency, and marginal cost).
3. Evaluate a small portfolio spanning hosted and self-hosted candidates.
4. Eliminate candidates that violate hard constraints; retain artifacts for later regression testing.

#### 1.4.1 RA Evaluation Harness Design

To perform a baseline sweep for the RA, we need a test harness that simulates real research queries and measures model performance against our constraints.

*Test Set Construction:*
100 questions about academic papers, stratified by:
- **Query type:** Factual (40%), analytical (30%), synthesis (20%), procedural (10%)
- **Paper domain:** CS/ML (50%), biology (20%), physics (15%), social sciences (15%)
- **Complexity:** Simple (1 paper, 1 claim) to complex (multiple papers, conflicting findings)

*Example Questions:*

1. **Factual (simple):**
   > "What dataset did Vaswani et al. (2017) use to evaluate the Transformer?"
   - Ground truth: WMT 2014 English-German, WMT 2014 English-French
   - Citation: Vaswani et al., "Attention is All You Need", NeurIPS 2017

2. **Analytical (medium):**
   > "What are the main limitations of BERT's pre-training approach according to recent critiques?"
   - Ground truth: Requires annotators to extract claims from 3-5 papers (e.g., Liu et al. 2019 RoBERTa, Clark et al. 2020 ELECTRA)
   - Expected citations: Multiple papers with specific section/page references

3. **Synthesis (complex):**
   > "How do the findings of Devlin et al. (2019) on masked language modeling compare to the critiques raised in subsequent work?"
   - Ground truth: Requires comparing BERT paper with RoBERTa, ELECTRA, and other follow-ups
   - Expected output: Structured comparison with multiple citations

#### 1.4.2 Baseline Sweep Results (RA Example)

Running the evaluation harness on GPT-5.2, Sonnet 4.5, and Llama 4 Maverick over 100 test questions:

| Model | Citation Precision | Claim Support | Tool-Call Success | p95 Latency | Cost/Query |
|-------|-------------------|---------------|-------------------|-------------|------------|
| **GPT-5.2** | 0.91 | 0.89 | 0.94 | 3.8s | $0.08 |
| **Sonnet 4.5** | 0.88 | 0.86 | 0.91 | 3.1s | $0.06 |
| **Llama 4 Maverick** | 0.84 | 0.82 | 0.85 | 4.2s | $0.02* |

*Constraint Evaluation:*

Hard constraints (must meet all):
- Citation precision ≥0.85: ✅ GPT-5.2, ✅ Sonnet 4.5, ❌ Llama 4 (0.84)
- p95 latency ≤5.0s: ✅ All models
- Cost/query ≤$0.15: ✅ All models

**Llama 4 Maverick narrowly eliminated** — violates citation precision hard constraint (0.84 < 0.85). Could be reconsidered with fine-tuning on academic citation format, or the constraint threshold could be revisited.

**Remaining candidates:** GPT-5.2, Sonnet 4.5

### 1.5 Defensible selection framework

Shortlisting reduces the model search space but does not determine a final choice.
A defensible selection framework makes model choice explicit, repeatable, and auditable, and reduces the risk that selection is driven by subjective demonstrations.

#### 1.5.1 Hard constraints as decision gates

Hard constraints define feasibility.
Any candidate violating at least one hard constraint is excluded from further consideration, even if it scores highly on other criteria.
This "gate first" structure prevents averaging from masking unacceptable failures.

#### 1.5.2 Weighted scoring among feasible candidates

Among feasible candidates, a weighted scoring model can be used to encode explicit product priorities.
The purpose of the scoring model is not mathematical sophistication but transparency and reproducibility.

**Criteria and weights.**
Criteria should be derived from the constraint specification.
For evidence-grounded agent products (including an academic research assistant), evidence quality and tool-use reliability are typically first-order.

Example criteria and weights:
- Evidence quality: 0.35
- Tool-use reliability: 0.20
- Latency: 0.20
- Cost: 0.15
- Disambiguation behavior: 0.10

**Metric normalization.**
Metrics should be normalized to a comparable 0–1 scale with respect to budgets.
For budgeted quantities such as latency and cost, a simple normalization is:

```text
normalized_latency = max(0, 1 - (latency / latency_budget))
normalized_cost    = max(0, 1 - (cost / cost_budget))
```

Evidence-related metrics (e.g., claim-support rate, citation precision) may already lie in [0, 1] under a defined rubric.

**Score computation.**

```text
score = Σ_i w_i * metric_i
```

The scoring stage should not be used to compensate for violations of hard constraints.

#### 1.5.3 Decision log and re-evaluation

A selection decision should be recorded in a short decision log that includes:

- the constraint specification (hard/soft)
- candidate set
- evaluation artifacts (inputs, outputs, measurements)
- the scoring rule and weights
- the selected model and rationale

The decision should be revisited when constraints change (e.g., a new cost ceiling) or when model/provider characteristics change.

### 1.6 Chapter remainder (outline)

The remainder of this chapter is intentionally retained as an outline and will be expanded in later work.

- Failure modes and mitigation strategies (outline)
- Production readiness considerations (outline)
- Worked decision matrix example for the RA (outline)
- Sensitivity analysis (outline)
- Checklist (outline)

### References

- Beyer, B., Jones, C., Petoff, J., & Murphy, N. R. (Eds.). (2016). *Site Reliability Engineering: How Google Runs Production Systems*. O'Reilly Media.
- Jain, R. (1991). *The Art of Computer Systems Performance Analysis: Techniques for Experimental Design, Measurement, Simulation, and Modeling*. Wiley.
- Keeney, R. L., & Raiffa, H. (1993). *Decisions with Multiple Objectives: Preferences and Value Tradeoffs*. Cambridge University Press.
- Lewis, P., Perez, E., Piktus, A., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS*.
- Rajpurkar, P., Jia, R., & Liang, P. (2018). Know What You Don't Know: Unanswerable Questions for SQuAD. *ACL*.
- Vaswani, A., et al. (2017). Attention is All You Need. *NeurIPS*.
- Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *NAACL*.
- Liu, Y., et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach. *arXiv*.
- Clark, K., et al. (2020). ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators. *ICLR*.

---

## Chapter 2: Fine-tuning

*(To be drafted)*

This chapter will cover when and how to fine-tune models for domain-specific performance, including:
- When fine-tuning is worth the investment vs. prompt engineering
- Data requirements and curation for fine-tuning
- Fine-tuning strategies (full, LoRA, adapters)
- Evaluation of fine-tuned models against baselines
- RA example: fine-tuning for academic citation format

---

## Chapter 3: Evaluation

*(To be drafted)*

This chapter will cover building robust evaluation harnesses for AI products, including:
- Evaluation dataset construction and maintenance
- Metrics selection and implementation
- Human evaluation protocols
- Automated evaluation pipelines
- Regression testing and monitoring
- RA example: citation precision and claim-support evaluation

---

## Chapter 4: Agent Architecture

*(To be drafted)*

This chapter will cover designing the internal logic and workflow of AI agents, including:
- Agent loop design (observe-think-act patterns)
- Tool integration and orchestration
- Context management and memory
- Error handling and recovery
- Multi-agent coordination
- RA example: paper search → retrieval → synthesis → citation workflow
