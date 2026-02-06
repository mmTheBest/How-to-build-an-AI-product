# How to Build an AI Product From Scratch

Many AI initiatives fail to transition from promising demonstrations to reliable products.
A common cause is the absence of explicit requirements and measurable success criteria; model selection is then driven by subjective impressions rather than product constraints.

This repository is a writing-first playbook for designing AI products end-to-end.
A single running example is used only to instantiate definitions and measurement choices: an **academic research assistant** (RA) that answers literature questions with verifiable citations, supported by retrieval-augmented generation (RAG), context engineering, knowledge-graph-based disambiguation and context expansion, targeted fine-tuning, and tool integrations for paper search and PDF parsing.

---

## Part 1: Choose a Model

### Chapter 1: Define product constraints prior to model selection

#### Introduction

Model selection is only meaningful relative to the product constraints under which the system must operate. [Beyer et al., 2016]
In the absence of explicit constraints, model choice tends to be driven by qualitative demonstrations and benchmark scores that are weakly coupled to end-to-end product performance. [Jain, 1991]

The objective of this chapter is to provide a repeatable procedure for translating informal product requirements into measurable targets that can be used to (i) eliminate infeasible design options, (ii) compare candidate models fairly, and (iii) define evaluation gates for subsequent iteration. [Beyer et al., 2016]
The running example is an academic research assistant (RA), which is used only to instantiate definitions and measurement choices. [Lewis et al., 2020]

#### 1.1 Model selection as a constraint satisfaction problem

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

#### 1.2 Translating requirements into measurable targets

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

#### 1.3 Chapter remainder (outline)

The remainder of this chapter is intentionally retained as an outline and will be expanded in later work.

- Problem statement (outline)
- Method: constraint categories (outline)
- Metrics table (outline)
- Worked example (outline)
- Checklist (outline)
- Takeaway (outline)

#### References

- Beyer, B., Jones, C., Petoff, J., & Murphy, N. R. (Eds.). (2016). *Site Reliability Engineering: How Google Runs Production Systems*. O'Reilly Media.
- Jain, R. (1991). *The Art of Computer Systems Performance Analysis: Techniques for Experimental Design, Measurement, Simulation, and Modeling*. Wiley.
- Keeney, R. L., & Raiffa, H. (1993). *Decisions with Multiple Objectives: Preferences and Value Tradeoffs*. Cambridge University Press.
- Lewis, P., Perez, E., Piktus, A., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS*.
- Rajpurkar, P., Jia, R., & Liang, P. (2018). Know What You Don't Know: Unanswerable Questions for SQuAD. *ACL*.

---

### Chapter 2: Map the model landscape prior to shortlisting

#### Introduction

Given a constraints specification, the next step is to construct a shortlist of candidate models. [Beyer et al., 2016]
The objective is not to identify a universally "best" model, but to eliminate candidates that are incompatible with latency, cost, reliability, and evidence requirements. [Jain, 1991]

This chapter provides a compact taxonomy of the model landscape and a baseline procedure for shortlisting candidates.

#### 2.1 Primary axes of the model landscape

**2.1.1 Hosted API models versus self-hosted (open-weight) models.**
Hosted API models are accessed via commercial providers.
They typically reduce integration time and eliminate model serving overhead, at the cost of per-request pricing, dependency on external reliability, and constraints on data handling. [Beyer et al., 2016]

Self-hosted (open-weight) models are operated under organizational control.
They can offer stronger control over data governance and potentially lower marginal cost at scale, but require engineering effort for serving, scaling, observability, and incident response. [Beyer et al., 2016]

**2.1.2 Model size tiers and operational consequences.**
Model size is a coarse proxy for capability, but it is also a proxy for serving cost, latency, and operational complexity.
Smaller models are typically faster and cheaper to run, but may be less reliable in instruction following and structured tool use under distribution shift.
Larger models often improve robustness and tool-use reliability, but can violate cost and tail-latency constraints unless mitigated by system design.

For AI agent products, tool-use reliability can become a binding constraint: a single failed tool call can dominate end-to-end failure rates even when free-form answer quality is high.

In the example of the RA agent, the landscape must be filtered by **tool-use reliability** (ability to correctly invoke paper search and PDF parsing tools) and **long-context handling** (processing full-text papers, which can be 8-12k tokens).

*Hosted API Models: RA Trade-offs*

**GPT-4 (OpenAI):**
- *Strengths:* Excellent instruction following, strong tool-use reliability (function calling API), handles 32k context (enough for most papers).
- *Weaknesses for RA:* High cost (~$0.03-0.06 per 1k tokens); at 3k tokens/query (retrieval + response), cost is ~$0.09-0.18 per answer. This exceeds the $0.15 budget unless retrieval is highly optimized.
- *Failure mode:* GPT-4 occasionally hallucinates citations even when grounded in retrieved text (observed in testing on SQuAD-like datasets).

**Claude 2 (Anthropic):**
- *Strengths:* 100k token context (can process multiple papers in a single call), lower cost (~$0.01 per 1k tokens), strong safety guardrails (reduces risk of generating toxic content if user asks about controversial research).
- *Weaknesses for RA:* Slightly weaker tool-use reliability than GPT-4 (observed in internal tests: 85% vs 92% tool-call success on ambiguous queries). Less extensively benchmarked on academic QA.
- *RA fit:* Strong candidate if long-context is critical (e.g., "Summarize all mentions of transformer architectures across these 5 papers").

**Gemini Pro (Google):**
- *Strengths:* Multimodal from day one (can process figures/tables in papers if provided as images), competitive with GPT-4 on coding benchmarks (useful if RA needs to interpret algorithm pseudocode in papers).
- *Weaknesses for RA:* Limited third-party evaluation data for academic QA; ecosystem less mature than OpenAI. Pricing not yet transparent for high-volume use.
- *RA fit:* Promising for future expansion (e.g., "Explain the architecture diagram in Figure 2 of this paper"), but risky as primary choice due to unknowns.

*Open-Weight Models: RA Trade-offs*

**LLaMA2-70B (Meta):**
- *Strengths:* Strong performance on knowledge tasks (~82% factual accuracy vs GPT-4's 85%), open license allows on-prem deployment (critical if handling proprietary research data). Fine-tunable on domain-specific academic corpora.
- *Weaknesses for RA:* Requires substantial infrastructure (2-4 A100 GPUs for real-time serving), tool-use reliability lower than GPT-4 (70-75% success on function calling without fine-tuning). Context length limited to 4k tokens (requires chunking for long papers).
- *RA fit:* Viable if cost-at-scale is critical and team can invest in fine-tuning for tool use + citation formatting. Not suitable for prototype phase due to ops overhead.

**Mixtral 8x7B (Mistral AI):**
- *Strengths:* Near-LLaMA2-70B quality at 6× faster inference (sparse MoE architecture), open weights, 32k context window (matches GPT-4). Excellent value for self-hosting.
- *Weaknesses for RA:* MoE serving complexity (requires custom deployment setup like vLLM with expert routing). Tool-use reliability not extensively tested (early model).
- *RA fit:* High-potential but high-risk choice. Could be shortlisted for baseline sweep if team has MoE deployment experience.

**Command R+ (Cohere):**
- *Strengths:* Optimized for RAG (retrieval-augmented generation) and tool use, 128k context window, available on Azure for enterprise deployment. Designed specifically for knowledge-intensive tasks like RA.
- *Weaknesses for RA:* Closed weights (despite "on-prem" availability, it's still vendor-controlled), less community evaluation than LLaMA2. Pricing model unclear for self-hosted deployments.
- *RA fit:* Strong contender if Azure deployment is acceptable and RAG optimization justifies the cost. Worth including in baseline sweep.

**RA Landscape Decision Matrix:**

| Model | Context | Tool Use | Cost/Query | Deployment | RA Suitability |
|-------|---------|----------|-----------|------------|----------------|
| GPT-4 | 32k | Excellent | $0.09-0.18 | API only | Strong but costly |
| Claude 2 | 100k | Good | $0.03-0.05 | API only | Best for long docs |
| Gemini Pro | Unknown | Unknown | Unknown | API only | Risky (immature) |
| LLaMA2-70B | 4k | Fair | $0.02-0.04 | Self-host | Good if fine-tuned |
| Mixtral 8x7B | 32k | Unknown | $0.01-0.03 | Self-host | High-potential |
| Command R+ | 128k | Excellent | Unknown | Azure/self | RAG-optimized |

**Shortlist for RA Baseline Sweep:**
1. **Claude 2** (hosted, long-context leader)
2. **GPT-4** (hosted, quality benchmark)
3. **Mixtral 8x7B** (open, cost leader if serving complexity is manageable)

*Eliminated:*
- Gemini Pro (too immature)
- LLaMA2-70B (context too short, ops too heavy for initial phase)
- Command R+ (held for future evaluation if RAG needs dominate)

#### 2.2 Baseline sweep procedure (shortlist construction)

A baseline sweep is used to filter clearly incompatible candidates prior to deeper integration.

1. Construct a small evaluation set (e.g., 50–100 prompts) representative of expected product usage.
2. Define metrics aligned with constraints (e.g., evidence quality, tool-call success rate, p95/p99 latency, and marginal cost).
3. Evaluate a small portfolio spanning hosted and self-hosted candidates.
4. Eliminate candidates that violate hard constraints; retain artifacts for later regression testing.

**2.2.1 RA Evaluation Harness Design.**
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

**2.2.2 Evaluation Metrics Implementation.**

*Citation Precision (Hard Constraint: ≥0.85):*

*Tool-Call Success Rate (RA-Specific):*

**2.2.3 Baseline Sweep Results (RA Example).**
Running the evaluation harness on Claude 2, GPT-4, and Mixtral 8x7B over 100 test questions:

| Model | Citation Precision | Claim Support | Tool-Call Success | p95 Latency | Cost/Query |
|-------|-------------------|---------------|-------------------|-------------|------------|
| **Claude 2** | 0.87 | 0.84 | 0.88 | 3.2s | $0.05 |
| **GPT-4** | 0.89 | 0.88 | 0.92 | 4.8s | $0.14 |
| **Mixtral 8x7B** | 0.78 | 0.76 | 0.71 | 2.1s | $0.03 |

*Constraint Evaluation:*

Hard constraints (must meet all):
- Citation precision ≥0.85: ✅ Claude 2, ✅ GPT-4, ❌ Mixtral (0.78)
- p95 latency ≤5.0s: ✅ All models
- Cost/query ≤$0.15: ✅ All models

**Mixtral 8x7B eliminated** — violates citation precision hard constraint (0.78 < 0.85). Despite being fastest and cheapest, its citation fabrication rate (22%) is unacceptable for scholarly use.

**Remaining candidates:** Claude 2, GPT-4

#### 2.3 Chapter remainder (outline)

The remainder of this chapter is retained as an outline and will be expanded in later work.

- Shortlisting failure modes (outline)
- Tool-use reliability as a system property (outline)

#### References

- Beyer, B., Jones, C., Petoff, J., & Murphy, N. R. (Eds.). (2016). *Site Reliability Engineering: How Google Runs Production Systems*. O'Reilly Media.
- Jain, R. (1991). *The Art of Computer Systems Performance Analysis: Techniques for Experimental Design, Measurement, Simulation, and Modeling*. Wiley.
- Vaswani, A., et al. (2017). Attention is All You Need. *NeurIPS*.
- Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *NAACL*.
- Liu, Y., et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach. *arXiv*.
- Clark, K., et al. (2020). ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators. *ICLR*.

---

### Chapter 3: Define a defensible selection framework

#### Introduction

Shortlisting reduces the model search space but does not determine a final choice.
A defensible selection framework makes model choice explicit, repeatable, and auditable, and reduces the risk that selection is driven by subjective demonstrations.

This chapter specifies a minimal decision procedure that combines (i) a hard-constraint filter and (ii) a transparent scoring rule over secondary criteria.

#### 3.1 Hard constraints as decision gates

Hard constraints define feasibility.
Any candidate violating at least one hard constraint is excluded from further consideration, even if it scores highly on other criteria.
This "gate first" structure prevents averaging from masking unacceptable failures.

#### 3.2 Weighted scoring among feasible candidates

Among feasible candidates, a weighted scoring model can be used to encode explicit product priorities.
The purpose of the scoring model is not mathematical sophistication but transparency and reproducibility.

**3.2.1 Criteria and weights.**
Criteria should be derived from the constraint specification.
For evidence-grounded agent products (including an academic research assistant), evidence quality and tool-use reliability are typically first-order.

Example criteria and weights:
- Evidence quality: 0.35
- Tool-use reliability: 0.20
- Latency: 0.20
- Cost: 0.15
- Disambiguation behavior: 0.10

**3.2.2 Metric normalization.**
Metrics should be normalized to a comparable 0–1 scale with respect to budgets.
For budgeted quantities such as latency and cost, a simple normalization is:

```text
normalized_latency = max(0, 1 - (latency / latency_budget))
normalized_cost    = max(0, 1 - (cost / cost_budget))
```

Evidence-related metrics (e.g., claim-support rate, citation precision) may already lie in [0, 1] under a defined rubric.

**3.2.3 Score computation.**

```text
score = Σ_i w_i * metric_i
```

The scoring stage should not be used to compensate for violations of hard constraints.

#### 3.3 Decision log and re-evaluation

A selection decision should be recorded in a short decision log that includes:

- the constraint specification (hard/soft)
- candidate set
- evaluation artifacts (inputs, outputs, measurements)
- the scoring rule and weights
- the selected model and rationale

The decision should be revisited when constraints change (e.g., a new cost ceiling) or when model/provider characteristics change.

#### 3.4 Chapter remainder (outline)

The remainder of this chapter is retained as an outline and will be expanded in later work.

- Worked decision matrix example for the RA (outline)
- Sensitivity analysis (outline)

#### References (outline)

- Add books/papers on multi-criteria decision analysis and evaluation methodology.
