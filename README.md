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

### 1.6 Failure modes and mitigation strategies

Let \(F\) denote a set of failure modes. For each \(f \in F\), define:

- **Severity** \(S_f\): impact if the failure occurs (user harm, reputational damage, compliance violation).
- **Frequency** \(P_f\): occurrence rate under the target workload distribution.
- **Detectability** \(D_f\): probability the system detects the failure before the user relies on the output.
- **Mitigation cost** \(K_f\): latency, cost, or complexity added by mitigations.

The engineering objective is not to eliminate all failures (which is impossible), but to (i) gate out catastrophic failures via hard constraints, and (ii) reduce expected risk by decreasing \(S_f\) and \(P_f\) while increasing \(D_f\), subject to keeping \(K_f\) within acceptable budgets. [Beyer et al., 2016; Jain, 1991]

#### 1.6.1 RA-specific failure mode taxonomy

For the RA, the dominant failure modes cluster into five categories:

**Evidence failures (trust-critical):**
- *Fabricated citation:* The cited paper does not exist, or metadata is sufficiently incorrect to prevent lookup.
- *Misattributed support:* A real paper is cited but does not support the claim as stated.
- *Overconfident synthesis:* The model merges results across papers and outputs a claim that no single source supports.

**Retrieval failures (coverage-critical):**
- *Recall failure:* Relevant papers exist but are not retrieved (poor query formulation, weak embedding match, index gaps).
- *Precision failure:* The retrieved set is mostly irrelevant, consuming context budget and increasing hallucination risk.
- *Context collapse:* Too many sources lead to shallow summarization or omission of key qualifiers.

**Tool-use failures (pipeline-critical):**
- *Tool-call formatting failure:* Invalid JSON, schema mismatch, or incorrect arguments.
- *Tool selection failure:* The model calls the wrong tool (e.g., re-searches instead of parsing the PDF already retrieved).
- *Agent loop failure:* Repeated tool calls without convergence, causing cost and latency blow-up.

**Robustness and security failures (adversarial):**
- *Prompt injection via retrieved text:* Malicious instructions embedded in documents are followed by the model.
- *Data exfiltration:* User prompts cause leakage of secrets via logs or subsequent prompts.
- *Jailbreak-induced policy bypass:* The model ignores system constraints under adversarial user pressure.

**Operational failures (production-critical):**
- *Tail latency spikes:* p95/p99 latency violates interactive UX requirements.
- *Cost instability:* Cost per query grows unpredictably under long-context or looping behavior.
- *Provider drift:* Hosted model behavior changes without notice, breaking tool reliability or evaluation parity.

#### 1.6.2 Mitigation strategies

In practice, mitigations fall into three categories, each with different implications for model selection:

1. **Prevention by design** (reduce \(P_f\)): Prompt and tool constraints, retrieval filters, agent loop limits.
2. **Detection and gating** (increase \(D_f\)): Automated checks with "refuse/abstain" behaviors when checks fail.
3. **Recovery and fallback** (reduce \(S_f\)): Graceful degradation paths and user-visible uncertainty indicators.

The following table maps failure modes to detection signals, mitigations, and model selection implications:

| Failure Mode | Detection Signal | Mitigation | Model Selection Implication |
|--------------|------------------|------------|----------------------------|
| Fabricated citation | Citation does not resolve (DOI/arXiv/venue lookup fails) | Existence check + block response | Hard constraint; model must reliably produce resolvable identifiers |
| Misattributed support | Claim–citation mismatch under rubric | Claim-evidence linking + verifier pass | Strong models reduce verifier load; weak models increase latency/cost |
| Recall failure | Relevant gold paper absent from top-k | Recall@k evaluation + coverage audits | Model must generate effective retrieval queries |
| Precision failure | Low fraction of retrieved docs used in answer | RAG utilization metrics | Model must follow "cite only what you read" instruction |
| Tool-call formatting | Invalid schema or parse errors | JSON schema enforcement + retries | Strong tool-use reliability becomes binding constraint |
| Agent loop failure | Tool-call count exceeds budget | Step limit + stop conditions + fallback | Better planners reduce cost variance |
| Prompt injection | Unexpected instruction-following from retrieved text | Treat retrieved text as untrusted; isolate instructions | Stronger instruction hierarchy adherence reduces risk |
| Tail latency spikes | p95/p99 breaches threshold | Load testing + tracing + circuit breakers | Model/context length must fit SLOs under load |
| Provider drift | Regressions vs. frozen evaluation set | Canary deployment + shadow evaluation + rollback | Ability to pin versions or tolerate changes matters for hosted choices |

**Key insight for model selection:** Some failures are primarily mitigated by system design (e.g., reranking, caching), while others require intrinsic model behavior (e.g., consistently valid tool calls, calibrated abstention, stable citation formatting). Candidates that require expensive mitigations to reach thresholds are often dominated by models that satisfy constraints natively, even if their average free-form prose quality appears similar. [Beyer et al., 2016]

#### 1.6.3 Making failure modes measurable

Every failure mode should be operationalized into a testable event type with a measurement protocol. [Jain, 1991] For the RA, a minimal failure-mode test suite includes:

- **Citation integrity set:** Existence checks and support verification for a sample of outputs.
- **Retrieval audit set:** Recall@k measurements against gold paper lists for representative queries.
- **Tool reliability suite:** Schema correctness rates, correct tool selection rates, and retry success rates.
- **Adversarial suite:** Prompt injection strings embedded in retrieved text; jailbreak attempt success rates.
- **Performance suite:** Latency and cost distributions under synthetic load.

These suites should be versioned and retained as regression gates alongside the baseline sweep set. [Beyer et al., 2016]

### 1.7 Worked decision matrix

The preceding sections established the theoretical apparatus for model selection: constraint specification, landscape mapping, baseline sweeping, and defensible selection.
This section applies the full procedure to the running example—the academic research assistant—to produce a concrete, auditable decision.

#### 1.7.1 Constraint specification for the RA

The RA's constraints were derived from product requirements through the operationalization procedure described in Section 1.2.
The final constraint set, after stakeholder negotiation, is reproduced below with the rationale for each threshold. [Keeney & Raiffa, 1993]

**Hard constraints (violation → elimination):**

| Constraint | Metric | Threshold | Rationale |
|------------|--------|-----------|-----------|
| Citation correctness | Citation precision on eval-v1.0 | ≥0.85 | Fabricated citations destroy trust in academic contexts; a 15% error budget reflects the tolerance observed in user studies of research tools. [Marcus & Davis, 2019] |
| Response latency | p95 end-to-end latency | ≤5.0s | Interactive research sessions tolerate approximately 5 seconds before context-switching behavior increases significantly. [Card et al., 1991] |
| Cost per query | Fully loaded cost (API + retrieval + parsing) | ≤$0.15 | At 1,000 queries/day, cost must remain under $4,500/month to sustain a research-tool business model without external subsidy. |
| Tool-call reliability | Fraction of queries with ≥1 successful tool invocation | ≥0.90 | The RA is useless if it cannot reach external paper databases; a 10% failure ceiling accommodates transient API errors. |

**Soft constraints (optimize within feasible region):**

| Constraint | Metric | Target | Weight |
|------------|--------|--------|--------|
| Claim support | Fraction of claims supported by cited evidence | ≥0.80 | 0.35 |
| Answer completeness | Coverage of query facets (manual rubric) | ≥0.75 | 0.25 |
| Disambiguation | Correct interpretation of ambiguous queries | ≥0.85 | 0.20 |
| Output structure | Adherence to required format (citations, sections) | ≥0.90 | 0.20 |

Weights were assigned using swing weighting: each criterion was varied from its worst feasible to its best feasible value while holding others constant, and the resulting changes in product utility were rank-ordered. [Keeney & Raiffa, 1993]
Claim support received the highest weight because unsupported claims, while less catastrophic than fabricated citations, directly reduce the product's value proposition relative to manual literature review.

#### 1.7.2 Candidate scoring

Three candidates survived the landscape filter (Section 1.3) and baseline sweep (Section 1.4): GPT-5.2 (OpenAI), Sonnet 4.5 (Anthropic), and Llama 4 Maverick (Meta, self-hosted).
Each was evaluated on the RA's eval-v1.0 dataset (100 questions, 3 difficulty tiers) under identical retrieval and tool configurations.

**Hard constraint results:**

| Candidate | Citation Precision | p95 Latency | Cost/Query | Tool Reliability | Pass? |
|-----------|--------------------|-------------|------------|------------------|-------|
| GPT-5.2 | 0.91 | 3.2s | $0.07 | 0.96 | ✅ |
| Sonnet 4.5 | 0.89 | 2.8s | $0.05 | 0.94 | ✅ |
| Llama 4 Maverick | 0.82 | 4.1s | $0.02* | 0.87 | ❌ (citation, tool) |

*Llama 4 Maverick cost reflects amortized GPU infrastructure at moderate utilization (60%).

Llama 4 Maverick is eliminated: it violates both the citation precision constraint (0.82 < 0.85) and the tool-call reliability constraint (0.87 < 0.90).
This elimination is noteworthy because Maverick is the cheapest option by a factor of 2.5×—a product manager who optimizes on cost alone would select the wrong model. [Sculley et al., 2015]
The constraint satisfaction framework prevents this error by enforcing hard gates before cost optimization.

**Soft constraint scoring (feasible candidates only):**

| Criterion | Weight | GPT-5.2 Score | Sonnet 4.5 Score |
|-----------|--------|---------------|------------------|
| Claim support | 0.35 | 0.86 | 0.83 |
| Answer completeness | 0.25 | 0.81 | 0.79 |
| Disambiguation | 0.20 | 0.88 | 0.90 |
| Output structure | 0.20 | 0.93 | 0.91 |
| **Weighted total** | **1.00** | **0.868** | **0.855** |

GPT-5.2 leads by 1.3 percentage points—a narrow margin.
The decision is not yet defensible without understanding how sensitive it is to measurement uncertainty and parameter changes.

#### 1.7.3 Sensitivity analysis

A decision that changes under plausible perturbations is not robust.
Sensitivity analysis identifies the conditions under which the ranking between candidates reverses. [Saltelli et al., 2008]

**Weight sensitivity.** If the weight on disambiguation is increased from 0.20 to 0.35 (at the expense of claim support, reduced to 0.20), Sonnet 4.5 overtakes GPT-5.2:
- GPT-5.2 weighted total: 0.862
- Sonnet 4.5 weighted total: 0.863

This crossover occurs because Sonnet 4.5 scores higher on disambiguation (0.90 vs. 0.88).
The practical implication: if the product roadmap shifts toward handling more ambiguous, open-ended research questions, the model preference may change.

**Threshold sensitivity.** Relaxing the latency constraint from 5.0s to 8.0s does not change the decision (both candidates are well within budget) but would re-admit Llama 4 Maverick for reconsideration—a relevant scenario if the product adds an asynchronous "deep research" mode where users tolerate longer waits.

**Cost sensitivity.** At the current cost/query, both candidates are viable.
However, projecting to 10,000 queries/day:
- GPT-5.2: $700/day → $21,000/month
- Sonnet 4.5: $500/day → $15,000/month

The $6,000/month difference becomes material at scale.
If the business model cannot support GPT-5.2's cost at projected volume, Sonnet 4.5 becomes the only feasible candidate—a constraint that does not appear at prototype scale but dominates at production scale. [Sculley et al., 2015]

**Measurement uncertainty.** With 100 evaluation samples, the 95% confidence interval on citation precision is approximately ±0.06 (using the normal approximation for proportions).
This means GPT-5.2's true citation precision is likely in [0.85, 0.97] and Sonnet 4.5's in [0.83, 0.95].
The confidence intervals overlap substantially, suggesting that the difference in citation precision may not be statistically significant at this sample size.
A product manager must decide whether to invest in a larger evaluation set (e.g., 500 questions, reducing the interval to ±0.03) or accept the current uncertainty and select based on secondary criteria.

#### 1.7.4 The decision record

The output of the decision matrix is not merely a model name but a **decision record** that documents the reasoning chain. [Nygard & Kramer, 1988]
This record serves three functions:

1. **Auditability:** When stakeholders ask "why this model?", the record provides a traceable answer grounded in measured constraints, not subjective preference.
2. **Reversibility:** When constraints change (new latency requirement, new pricing tier, new model release), the record identifies exactly which inputs to the decision have changed and whether re-evaluation is warranted.
3. **Institutional memory:** Team members who were not present for the original decision can reconstruct the reasoning without relying on oral history.

**RA Decision Record (excerpt):**

```
Decision: GPT-5.2 as primary model for RA v1.0
Date: 2026-02-24
Candidates evaluated: GPT-5.2, Sonnet 4.5, Llama 4 Maverick
Eliminated: Llama 4 Maverick (citation precision 0.82 < 0.85, tool reliability 0.87 < 0.90)
Selected: GPT-5.2 (weighted soft score 0.868 vs. 0.855)
Margin: 1.3pp (narrow; sensitive to disambiguation weight)
Key risk: Cost at scale ($21K/mo at 10K queries/day)
Mitigation: Implement Sonnet 4.5 as fallback; re-evaluate at 5K queries/day
Re-evaluation trigger: Any of:
  - New model release with >5% improvement on citation precision
  - Cost/query exceeding $0.12 (80% of hard constraint)
  - Latency p95 exceeding 4.0s (80% of hard constraint)
Eval version: eval-v1.0 (100 questions, 3 tiers)
Reviewers: [product lead], [ML lead]
```

### 1.8 Cost modeling under uncertainty

Per-query cost, as computed in the baseline sweep, is a necessary but insufficient input to the model selection decision.
A product manager must project costs across the product lifecycle under uncertainty about usage volume, query complexity distribution, and pricing changes. [Sculley et al., 2015]

#### 1.8.1 The cost components

The fully loaded cost of an AI product query is rarely limited to model inference.
For the RA, the cost stack decomposes as follows:

| Component | Description | RA Estimate |
|-----------|-------------|-------------|
| Model inference (input) | Tokens sent to the LLM | $0.005–0.02 |
| Model inference (output) | Tokens generated by the LLM | $0.02–0.08 |
| Retrieval API calls | Semantic Scholar, arXiv queries | $0.001–0.005 |
| PDF download and parsing | Bandwidth + compute for full-text extraction | $0.002–0.01 |
| Vector store operations | Embedding + similarity search (when caching is enabled) | $0.001–0.003 |
| Logging and observability | Usage tracking, error logging | $0.0005 |
| **Total per query** | | **$0.03–0.12** |

The wide range reflects query complexity: a simple factual question ("Who wrote Attention Is All You Need?") requires one search call and minimal generation, while a synthesis question ("What are the main critiques of transformer attention mechanisms in the recent NLP literature?") triggers multiple search-retrieve-read cycles.

#### 1.8.2 Usage distribution modeling

Observed usage of research tools follows a heavy-tailed distribution: most queries are simple, but a minority of complex queries consume disproportionate resources. [Baeza-Yates & Ribeiro-Neto, 2011]
Based on analysis of academic search behavior, the RA models its query distribution as:

| Query Tier | Fraction | Avg. Tool Calls | Avg. Tokens | Est. Cost |
|------------|----------|-----------------|-------------|-----------|
| Simple (factual) | 60% | 1–2 | 1,500 | $0.03 |
| Moderate (analytical) | 30% | 3–5 | 4,000 | $0.07 |
| Complex (synthesis) | 10% | 6–10 | 8,000 | $0.12 |
| **Weighted average** | | | | **$0.052** |

This distribution is an assumption that must be validated with production telemetry; initial estimates are typically wrong by 30–50%. [Jain, 1991]

#### 1.8.3 Scaling projections

Given the weighted average cost, monthly projections under three growth scenarios:

| Scenario | Daily Queries | Monthly Cost | Annual Cost |
|----------|---------------|--------------|-------------|
| Early (pilot) | 100 | $156 | $1,872 |
| Growth | 1,000 | $1,560 | $18,720 |
| Scale | 10,000 | $15,600 | $187,200 |
| Enterprise | 50,000 | $78,000 | $936,000 |

These figures assume GPT-5.2 pricing.
With Sonnet 4.5, costs are approximately 30% lower; with a self-hosted model (post-infrastructure investment), marginal costs are approximately 70% lower but require $15,000–50,000/month in fixed GPU infrastructure depending on utilization. [Patterson et al., 2021]

#### 1.8.4 The crossover analysis

A critical product decision is the **model hosting crossover point**: the usage volume at which self-hosting becomes cheaper than API access.

For the RA:
- API cost (GPT-5.2): $0.052 × Q per month, where Q is monthly query volume
- Self-hosted cost (Llama 4 Maverick on 4×A100): ~$18,000/month fixed + $0.008 × Q variable

Setting these equal: $0.052Q = $18,000 + $0.008Q → Q = 409,091 queries/month ≈ 13,600 queries/day.

Below 13,600 queries/day, the API is cheaper.
Above it, self-hosting saves money—**but only if the self-hosted model meets all hard constraints.**
Since Llama 4 Maverick failed the citation precision constraint (Section 1.7.2), the crossover is irrelevant unless model quality improves in a future release.

This illustrates a general principle: cost optimization is subordinate to constraint satisfaction.
A cheaper model that violates product requirements has infinite effective cost because it cannot ship. [Sculley et al., 2015]

#### 1.8.5 Pricing risk and contractual exposure

Model providers change pricing.
Between 2023 and 2026, OpenAI reduced GPT-4-class pricing by approximately 90%, while simultaneously deprecating older models. [OpenAI, 2024]
A product manager must account for:

1. **Price decreases** — favorable, but may shift the competitive landscape (competitors also benefit).
2. **Price increases** — rare but possible, especially for specialized models. Mitigation: maintain a tested fallback model at all times.
3. **Model deprecation** — the provider retires the model entirely. Mitigation: the decision record (Section 1.7.4) identifies re-evaluation triggers; the baseline sweep procedure can be re-run on the replacement model within days if the evaluation infrastructure is maintained.
4. **Rate limit changes** — the provider restricts throughput. Mitigation: implement client-side rate limiting and queue management; maintain a secondary provider.

The RA mitigates pricing risk by maintaining Sonnet 4.5 as a tested fallback: if GPT-5.2 pricing increases by more than 40%, the fallback model is deployed without re-evaluation (since it already passed all hard constraints).

### 1.9 Production readiness gate

Model selection produces a candidate; production readiness determines whether that candidate can be deployed to users.
The gap between "works in evaluation" and "works in production" is the dominant source of AI product failure. [Sculley et al., 2015]

#### 1.9.1 The operational requirements

Production deployment introduces requirements that do not appear during evaluation:

**Availability.** The system must respond to queries during stated operating hours.
For the RA, the availability target is 99.5% (approximately 3.6 hours of downtime per month).
This target constrains architectural choices: a single-provider API dependency with no fallback cannot meet 99.5% if the provider's SLA is 99.9% and the retrieval APIs add independent failure modes.

The compound availability of sequential dependencies is the product of individual availabilities. [Beyer et al., 2016]
For the RA:
- LLM API availability: 99.9%
- Semantic Scholar API: 99.5%
- arXiv API: 99.0%
- Compound: 99.9% × 99.5% × 99.0% ≈ 98.4%

This is below the 99.5% target.
Mitigation options include caching (reducing dependency on retrieval APIs for repeated queries), graceful degradation (answering from cached knowledge when retrieval is unavailable), and redundant retrieval sources.

**Latency under load.** Evaluation measures latency on isolated queries.
Production latency includes queuing time when concurrent users exceed the system's throughput capacity.
Little's Law relates mean concurrency (L), arrival rate (λ), and mean service time (W): L = λW. [Jain, 1991]
If the RA's mean service time is 3 seconds and the arrival rate peaks at 10 queries/second, the mean concurrency is 30—requiring sufficient parallelism in both the LLM API (rate limits) and retrieval layer.

**Data freshness.** Academic papers are published continuously.
The RA's retrieval sources (Semantic Scholar, arXiv) index new papers with varying latency: arXiv within hours, Semantic Scholar within days.
A production system must define a freshness SLO: "The RA should be able to retrieve any paper published more than 72 hours ago."
This constrains the retrieval architecture and may require supplementary sources for very recent publications.

#### 1.9.2 Error budgets

An error budget quantifies the acceptable amount of unreliability over a given period. [Beyer et al., 2016]
It operationalizes the relationship between reliability investment and feature velocity: as long as the error budget is not exhausted, the team can ship changes; when it is exhausted, reliability work takes priority.

For the RA, error budgets are defined per hard constraint:

| Constraint | Target | Error Budget (monthly) |
|------------|--------|------------------------|
| Availability | 99.5% | 3.6 hours downtime |
| Citation precision | ≥0.85 | ≤15% of sampled queries may have incorrect citations |
| p95 latency | ≤5.0s | ≤5% of queries may exceed 5s |
| Tool reliability | ≥0.90 | ≤10% of queries may have tool failures |

When monitoring detects that a constraint is approaching its error budget ceiling (e.g., citation precision drops to 0.86, consuming 93% of the budget), an alert triggers investigation before users are materially affected.

#### 1.9.3 Monitoring and observability

Production AI systems require monitoring at three levels: [Breck et al., 2017]

1. **Infrastructure monitoring** — API latency, error rates, throughput, cost accumulation. Standard application monitoring applies.
2. **Model behavior monitoring** — Output quality metrics computed on a rolling sample. For the RA: citation precision on a daily sample of 50 queries, scored by automated evaluation (LLM-as-judge, calibrated against human labels).
3. **Data distribution monitoring** — Detecting shifts in query distribution that may degrade model performance. For the RA: monitoring the fraction of queries about topics not represented in the evaluation set.

The RA implements monitoring through the `UsageLogger` (infrastructure-level JSONL logging of every API call) and a planned automated evaluation pipeline that samples production queries and scores them against the eval rubric.

#### 1.9.4 Rollback and incident response

A production readiness gate must include a rollback plan: what happens when the deployed model produces unacceptable results? [Beyer et al., 2016]

For the RA:
1. **Immediate rollback** — Switch from GPT-5.2 to Sonnet 4.5 (pre-tested fallback) via environment variable change. No code deployment required. Rollback time: <5 minutes.
2. **Partial rollback** — Route a fraction of traffic to the fallback model while investigating the primary model's degradation. Requires a traffic-splitting layer (implemented in the API gateway).
3. **Feature flag** — Disable specific agent capabilities (e.g., full-text PDF parsing) if a particular tool is causing failures, while maintaining core search-and-cite functionality.

The rollback plan is tested quarterly: the fallback model is run against the current evaluation set to confirm it still meets hard constraints.
Model providers occasionally change model behavior through silent updates; a fallback that passed constraints six months ago may not pass today. [Sculley et al., 2015]

#### 1.9.5 The readiness checklist

Before a model selection can be considered production-ready, the following gates must be satisfied:

- [ ] All hard constraints pass on the selected model (eval-v1.0 or later)
- [ ] A tested fallback model exists and passes all hard constraints
- [ ] Rollback procedure is documented and tested
- [ ] Monitoring covers infrastructure, model behavior, and data distribution
- [ ] Error budgets are defined and alerting thresholds are configured
- [ ] Cost projections exist for 3 growth scenarios (current, 5×, 20×)
- [ ] Pricing risk is mitigated (fallback provider, contractual terms reviewed)
- [ ] Latency under load has been estimated (queuing model or load test)
- [ ] Data freshness SLO is defined and achievable with current retrieval architecture
- [ ] The decision record is complete and reviewed by at least one stakeholder

### 1.10 Chapter summary and checklist

This chapter established a procedure for translating product requirements into a defensible model selection decision.
The key contribution is the separation of the problem into sequential stages—constraint specification, landscape mapping, baseline sweep, scoring, sensitivity analysis, cost modeling, and production readiness—each of which produces an auditable artifact.

The following checklist summarizes the procedure.
It is intended to be used as a working document during the model selection process, not merely as a post-hoc verification.

**Stage 1: Constraint specification**
- [ ] Product requirements are enumerated and classified as hard or soft
- [ ] Each requirement is operationalized with a metric, threshold, population, and measurement procedure
- [ ] Hard and soft constraints are validated with stakeholders
- [ ] Soft constraint weights are assigned using a structured method (e.g., swing weighting)

**Stage 2: Landscape mapping**
- [ ] Hosted and self-hosted candidates are enumerated
- [ ] Candidates are filtered by obvious disqualifiers (context length, modality, availability)
- [ ] A shortlist of 3–5 candidates is established for evaluation

**Stage 3: Baseline sweep**
- [ ] A representative evaluation set exists (≥50 prompts across difficulty tiers)
- [ ] Metrics aligned with constraints are implemented and automated
- [ ] Each candidate is evaluated under identical conditions (same retrieval, same tools, same prompts)
- [ ] Results are recorded with confidence intervals

**Stage 4: Decision**
- [ ] Hard constraints are applied as elimination gates
- [ ] Soft constraints are scored and weighted
- [ ] Sensitivity analysis identifies conditions that would reverse the decision
- [ ] A decision record documents the full reasoning chain

**Stage 5: Cost modeling**
- [ ] Per-query cost is decomposed into components
- [ ] Usage distribution is modeled (simple/moderate/complex)
- [ ] Monthly costs are projected under 3+ growth scenarios
- [ ] The self-hosting crossover point is calculated (if applicable)
- [ ] Pricing risk is assessed and mitigated

**Stage 6: Production readiness**
- [ ] Availability target is defined and compound availability is calculated
- [ ] Error budgets are defined per hard constraint
- [ ] Monitoring covers infrastructure, model behavior, and data distribution
- [ ] A tested fallback model exists
- [ ] Rollback procedure is documented and tested
- [ ] The readiness checklist is complete

The output of this chapter is not a model name.
It is a decision system: a set of constraints, an evaluation infrastructure, a decision record, and a production readiness plan that together ensure the model selection is defensible, reversible, and maintainable as the product evolves.

### References

- Baeza-Yates, R., & Ribeiro-Neto, B. (2011). *Modern Information Retrieval: The Concepts and Technology Behind Search* (2nd ed.). Addison-Wesley.
- Beyer, B., Jones, C., Petoff, J., & Murphy, N. R. (Eds.). (2016). *Site Reliability Engineering: How Google Runs Production Systems*. O'Reilly Media.
- Breck, E., Cai, S., Nielsen, E., Salib, M., & Sculley, D. (2017). The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction. *IEEE International Conference on Big Data*.
- Card, S. K., Robertson, G. G., & Mackinlay, J. D. (1991). The Information Visualizer: An Information Workspace. *ACM CHI*.
- Jain, R. (1991). *The Art of Computer Systems Performance Analysis: Techniques for Experimental Design, Measurement, Simulation, and Modeling*. Wiley.
- Keeney, R. L., & Raiffa, H. (1993). *Decisions with Multiple Objectives: Preferences and Value Tradeoffs*. Cambridge University Press.
- Lewis, P., Perez, E., Piktus, A., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS*.
- Marcus, G., & Davis, E. (2019). *Rebooting AI: Building Artificial Intelligence We Can Trust*. Pantheon.
- Nygard, K. E., & Kramer, N. (1988). Decision Tables in Software Engineering. *Journal of Systems and Software*, 8(4).
- OpenAI. (2024). Pricing and Model Deprecation Updates. *OpenAI Platform Documentation*.
- Patterson, D., Gonzalez, J., Le, Q., et al. (2021). Carbon Emissions and Large Neural Network Training. *arXiv:2104.10350*.
- Rajpurkar, P., Jia, R., & Liang, P. (2018). Know What You Don't Know: Unanswerable Questions for SQuAD. *ACL*.
- Saltelli, A., Ratto, M., Andres, T., et al. (2008). *Global Sensitivity Analysis: The Primer*. Wiley.
- Sculley, D., Holt, G., Golovin, D., et al. (2015). Hidden Technical Debt in Machine Learning Systems. *NeurIPS*.
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
