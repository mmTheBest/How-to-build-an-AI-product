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
The intent is not merely to arrive at a model name, but to demonstrate the reasoning chain in sufficient detail that a product manager facing a different AI product could adapt the procedure.

#### 1.7.1 Constraint specification for the RA

The RA's constraints were derived from product requirements through the operationalization procedure described in Section 1.2.
However, the operationalization process itself is rarely straightforward.
Before presenting the final constraint set, it is instructive to examine the negotiation process that produced it.

**The constraint negotiation process.**
Initial requirements gathering for the RA yielded statements such as "citations should be accurate," "the system should be fast," and "it shouldn't cost too much."
These statements are typical of early-stage product requirements: directionally correct but not actionable.
The operationalization procedure transforms them into measurable targets, but the specific thresholds require negotiation among stakeholders with competing priorities. [Keeney & Raiffa, 1993]

Consider citation precision.
The research team proposed a threshold of ≥0.95 (only 5% of citations may be incorrect), arguing that academic credibility demands near-perfection.
The engineering team countered that achieving 0.95 citation precision would require extensive post-generation verification (adding latency and cost), and that current state-of-the-art models achieve approximately 0.85–0.92 on comparable tasks without specialized pipelines.
The product manager mediated by asking a different question: *What is the user's tolerance for citation errors, given that the alternative (manual literature search) has its own error rate?*

Studies of manual literature review indicate that researchers miss approximately 20–30% of relevant sources and occasionally misattribute claims. [Bornmann & Mutz, 2015]
A citation precision of 0.85 means the RA makes citation errors roughly 3× less frequently than the failure modes it replaces—a meaningful improvement even if imperfect.
The threshold was set at 0.85 with a note that it should be revisited when the system matures and users develop higher expectations.

This negotiation pattern recurs across all constraints: the threshold is not derived from first principles alone, but from the intersection of technical feasibility, user tolerance, and competitive positioning.

**The final constraint set.**

Hard constraints (violation → elimination):

| Constraint | Metric | Threshold | Rationale |
|------------|--------|-----------|-----------|
| Citation correctness | Citation precision on eval-v1.0 | ≥0.85 | Fabricated citations destroy trust in academic contexts; the threshold reflects user tolerance calibrated against manual review error rates. [Marcus & Davis, 2019] |
| Response latency | p95 end-to-end latency | ≤5.0s | Interactive research sessions tolerate approximately 5 seconds before context-switching behavior increases significantly; this threshold is derived from human-computer interaction research on acceptable wait times for cognitively complex tasks. [Card et al., 1991] |
| Cost per query | Fully loaded cost (API + retrieval + parsing) | ≤$0.15 | At 1,000 queries/day, cost must remain under $4,500/month to sustain a research-tool business model without external subsidy. The threshold was back-calculated from a target gross margin of 60% at a $15/month subscription price with estimated 200 queries/user/month. |
| Tool-call reliability | Fraction of queries with ≥1 successful tool invocation | ≥0.90 | The RA is useless if it cannot reach external paper databases; a 10% failure ceiling accommodates transient API errors while ensuring that the vast majority of user queries receive grounded (not hallucinated) responses. |

The rationale column deserves emphasis.
A common failure mode in constraint specification is stating thresholds without recording *why* that specific number was chosen.
When thresholds lack documented rationale, they become arbitrary and resistant to principled revision.
A product manager who inherits a constraint of "≥0.85 citation precision" without context cannot determine whether it should be 0.80 or 0.90 when circumstances change. [Nygard & Kramer, 1988]

**Soft constraints (optimize within feasible region):**

| Constraint | Metric | Target | Weight | Rationale for Weight |
|------------|--------|--------|--------|---------------------|
| Claim support | Fraction of claims supported by cited evidence | ≥0.80 | 0.35 | Unsupported claims reduce value proposition vs. manual review; highest marginal impact on user trust after citation correctness |
| Answer completeness | Coverage of query facets (manual rubric) | ≥0.75 | 0.25 | Incomplete answers require follow-up queries, degrading user experience; second-order impact after trust |
| Disambiguation | Correct interpretation of ambiguous queries | ≥0.85 | 0.20 | Ambiguous queries represent ~15–20% of expected traffic; misinterpretation wastes retrieval budget and user time |
| Output structure | Adherence to required format (citations, sections) | ≥0.90 | 0.20 | Consistent formatting enables downstream integration (citation managers, note-taking tools); lower weight because formatting failures are less consequential than content failures |

**The swing weighting procedure.**
Weights were assigned using swing weighting, a structured method for eliciting preferences over multiple objectives. [Keeney & Raiffa, 1993]
The procedure is worth describing because it is commonly replaced by ad hoc weight assignment, which introduces undocumented bias.

1. **Define the swing range for each criterion.** For each soft constraint, identify the worst feasible value and the best feasible value:
   - Claim support: 0.60 (worst) to 0.95 (best)
   - Answer completeness: 0.50 (worst) to 0.90 (best)
   - Disambiguation: 0.70 (worst) to 0.95 (best)
   - Output structure: 0.75 (worst) to 0.98 (best)

2. **Rank the swings.** Ask: "If all criteria are at their worst value, which single criterion would provide the most value by swinging to its best value?" The answer reveals the most important criterion. For the RA, claim support was ranked first: a system that supports 95% of its claims with evidence is fundamentally more useful than one that supports only 60%, regardless of other criteria.

3. **Assign reference points.** The most important swing is assigned 100 points. Other swings are scored relative to it:
   - Claim support: 100 (reference)
   - Answer completeness: 71
   - Disambiguation: 57
   - Output structure: 57

4. **Normalize to weights.** Divide each score by the total: 100 + 71 + 57 + 57 = 285. Claim support: 100/285 ≈ 0.35, etc.

This procedure is transparent and reproducible: a different team performing the same elicitation may arrive at different weights, but the reasoning is traceable and debatable rather than opaque.

#### 1.7.2 Candidate scoring

Three candidates survived the landscape filter (Section 1.3) and baseline sweep (Section 1.4): GPT-5.2 (OpenAI), Sonnet 4.5 (Anthropic), and Llama 4 Maverick (Meta, self-hosted).
Each was evaluated on the RA's eval-v1.0 dataset (100 questions, 3 difficulty tiers) under identical retrieval and tool configurations.

**Controlling for confounders in model comparison.**
A subtle but critical requirement for fair model comparison is controlling for system-level confounders.
If model A is tested with a well-tuned prompt and model B with a generic prompt, the comparison measures prompt engineering skill, not model capability.
For the RA evaluation, the following were held constant across all candidates:

- Identical retrieval pipeline (same Semantic Scholar and arXiv clients, same rate limits)
- Identical tool definitions (same function signatures and descriptions)
- Identical system prompt (adapted only for model-specific formatting requirements, e.g., Claude's XML tags vs. OpenAI's function calling syntax)
- Identical evaluation set (eval-v1.0, 100 questions)
- Identical scoring rubric and scorer (same human annotator for claim support; same automated pipeline for citation precision)
- Identical hardware and network conditions (same host, same time of day to control for API load)

The only variable was the model itself.
This level of control is expensive—each candidate requires a full evaluation run costing approximately $5–15 in API fees plus 4–6 hours of human annotation.
But without it, the comparison is confounded and the decision is indefensible. [Jain, 1991]

**Hard constraint results:**

| Candidate | Citation Precision | p95 Latency | Cost/Query | Tool Reliability | Pass? |
|-----------|--------------------|-------------|------------|------------------|-------|
| GPT-5.2 | 0.91 | 3.2s | $0.07 | 0.96 | ✅ |
| Sonnet 4.5 | 0.89 | 2.8s | $0.05 | 0.94 | ✅ |
| Llama 4 Maverick | 0.82 | 4.1s | $0.02* | 0.87 | ❌ (citation, tool) |

*Llama 4 Maverick cost reflects amortized GPU infrastructure at moderate utilization (60%).

**Analysis of the elimination.**
Llama 4 Maverick is eliminated: it violates both the citation precision constraint (0.82 < 0.85) and the tool-call reliability constraint (0.87 < 0.90).

This elimination warrants careful examination because it illustrates several PM-relevant principles:

1. **Cost optimization is subordinate to constraint satisfaction.** Maverick is the cheapest option by a factor of 2.5×. A product manager who optimizes on cost alone—or who treats cost as a weighted criterion rather than gating on hard constraints first—would select a model that cannot ship. [Sculley et al., 2015]

2. **Multiple constraint violations compound risk.** Maverick fails on two independent constraints. Even if one violation were marginal (e.g., citation precision at 0.84 vs. threshold 0.85), the simultaneous failure on tool reliability indicates a systematic capability gap, not a measurement artifact.

3. **Elimination is not permanent.** The decision record should note the specific shortfall so that Maverick can be re-evaluated when Meta releases an improved version. If a future Maverick variant achieves 0.87 citation precision and 0.92 tool reliability, it becomes a serious contender given its cost advantage.

4. **Self-hosted models carry hidden costs that partially offset their price advantage.** The $0.02/query estimate assumes 60% GPU utilization on a dedicated cluster. At lower utilization (common during early adoption), the effective per-query cost increases. Infrastructure maintenance, model serving engineering, and on-call burden are not captured in the per-query number. A full total cost of ownership (TCO) comparison is required before concluding that self-hosting is cheaper.

**Soft constraint scoring (feasible candidates only):**

| Criterion | Weight | GPT-5.2 Score | Sonnet 4.5 Score |
|-----------|--------|---------------|------------------|
| Claim support | 0.35 | 0.86 | 0.83 |
| Answer completeness | 0.25 | 0.81 | 0.79 |
| Disambiguation | 0.20 | 0.88 | 0.90 |
| Output structure | 0.20 | 0.93 | 0.91 |
| **Weighted total** | **1.00** | **0.868** | **0.855** |

GPT-5.2 leads by 1.3 percentage points.
This margin is narrow, and the decision is not yet defensible without understanding its sensitivity to perturbations.

#### 1.7.3 Sensitivity analysis

A product decision that changes under plausible perturbations is not robust.
Sensitivity analysis identifies the conditions under which the ranking between candidates reverses, enabling the product manager to assess how much confidence to place in the decision and what future events might warrant re-evaluation. [Saltelli et al., 2008]

**Weight sensitivity.**
The most common source of decision instability is uncertainty in the weights assigned to soft constraints.
Swing weighting produces a single weight vector, but reasonable people might disagree on the relative importance of criteria.

If the weight on disambiguation is increased from 0.20 to 0.35 (at the expense of claim support, reduced to 0.20), the ranking reverses:
- GPT-5.2 weighted total: 0.862
- Sonnet 4.5 weighted total: 0.863

This crossover occurs because Sonnet 4.5 scores higher on disambiguation (0.90 vs. 0.88).
The practical implication is significant: if the product roadmap shifts toward handling more ambiguous, open-ended research questions (e.g., "What are the emerging critiques of large language models in the social sciences?"—a query requiring careful disambiguation of scope), the model preference may change.

A product manager should document this sensitivity explicitly:
*"The decision favors GPT-5.2 under the current weight vector, but is sensitive to the disambiguation weight. If product direction shifts toward open-ended queries, re-evaluate with adjusted weights."*

**Threshold sensitivity.**
Hard constraint thresholds define the boundary between acceptable and unacceptable.
Perturbing these boundaries reveals which constraints are "binding" (the candidate barely passes) and which have comfortable margin.

For GPT-5.2:
- Citation precision: 0.91 vs. threshold 0.85 → margin of 0.06 (comfortable)
- p95 latency: 3.2s vs. threshold 5.0s → margin of 1.8s (very comfortable)
- Cost/query: $0.07 vs. threshold $0.15 → margin of $0.08 (comfortable)
- Tool reliability: 0.96 vs. threshold 0.90 → margin of 0.06 (comfortable)

No hard constraint is binding for GPT-5.2.
This is favorable: it means minor degradation in model performance (due to API changes, traffic increases, or prompt modifications) is unlikely to trigger a constraint violation.

Contrast with a hypothetical candidate scoring 0.86 on citation precision.
With a margin of only 0.01 above the threshold, any measurement noise or production variance could push the metric below the hard gate.
A binding constraint increases operational risk and demands more frequent monitoring.

Relaxing the latency constraint from 5.0s to 8.0s would not change the ranking but would re-admit Llama 4 Maverick for consideration—a relevant scenario if the product adds an asynchronous "deep research" mode where users tolerate longer waits in exchange for more thorough analysis.
This kind of "what-if" analysis is essential for roadmap planning: it identifies which product decisions would change the model selection and which are model-neutral.

**Cost sensitivity at scale.**
At the current per-query cost, both candidates are well within the $0.15 hard constraint.
However, projecting to higher usage volumes reveals a divergence:

| Daily Queries | GPT-5.2 Monthly | Sonnet 4.5 Monthly | Delta |
|---------------|------------------|--------------------|-------|
| 100 | $210 | $150 | $60 |
| 1,000 | $2,100 | $1,500 | $600 |
| 10,000 | $21,000 | $15,000 | $6,000 |
| 50,000 | $105,000 | $75,000 | $30,000 |

At 10,000 queries/day, the $6,000/month difference is material—equivalent to a mid-level engineer's monthly cost.
At 50,000 queries/day, the $30,000/month difference may determine whether the product is profitable.

This analysis does not change the model selection for v1.0 (both candidates are within budget at launch volumes), but it establishes a **cost-triggered re-evaluation point**: when daily query volume reaches a level where the cost difference exceeds a meaningful fraction of operating budget, the product manager should re-run the decision matrix with updated weights that reflect the increased importance of cost.

**Measurement uncertainty.**
With 100 evaluation samples, statistical precision is limited.
The 95% confidence interval on citation precision (using the normal approximation for proportions) is approximately ±0.06.
This means GPT-5.2's true citation precision is likely in [0.85, 0.97] and Sonnet 4.5's in [0.83, 0.95].

The confidence intervals overlap substantially.
A product manager must confront this honestly: *the measured difference in citation precision between the two candidates is not statistically significant at this sample size.*
The decision therefore rests primarily on the soft constraint weighted total (where GPT-5.2 leads by 1.3pp) and on secondary considerations such as cost trajectory and provider relationship.

This has implications for evaluation investment: increasing the evaluation set to 500 questions would reduce the confidence interval to ±0.03, potentially resolving the ambiguity.
The cost of this investment (approximately $50–75 in API fees plus 20–30 hours of annotation) must be weighed against the value of a more confident decision.
For a v1.0 launch, the current ambiguity may be acceptable; for a production system processing thousands of queries daily, the investment in a larger evaluation set is almost certainly justified.

#### 1.7.4 The decision record

The output of the decision matrix is not merely a model name but a **decision record** that documents the full reasoning chain. [Nygard & Kramer, 1988]
This record is among the most important artifacts a product manager produces during model selection, yet it is frequently omitted in practice.

The decision record serves three functions:

1. **Auditability.** When a stakeholder, investor, or regulator asks "why this model?", the record provides a traceable answer grounded in measured constraints, not subjective preference. In regulated domains (healthcare, finance, government), this traceability may be legally required.

2. **Reversibility.** When conditions change—a new model is released, pricing changes, user behavior shifts, or a constraint is revised—the record identifies exactly which inputs to the decision have changed and whether re-evaluation is warranted. Without a record, every change triggers a full re-evaluation from scratch because no one remembers which factors were decisive.

3. **Institutional memory.** Team members who join after the original decision cannot reconstruct the reasoning from code or configuration alone. The record prevents the common failure mode where a successor product manager changes the model based on benchmark marketing without understanding the constraint-driven reasoning that selected the incumbent.

**RA Decision Record:**

```
Decision: GPT-5.2 as primary model for RA v1.0
Status: Approved
Date: 2026-02-24
Authors: [product lead], [ML lead]
Reviewers: [engineering lead], [research advisor]

CONTEXT
The RA requires a model capable of (a) reliable tool use for academic paper
retrieval, (b) accurate citation of sources, and (c) synthesis of findings
across multiple papers. The model must operate within defined cost and latency
envelopes.

CANDIDATES EVALUATED
1. GPT-5.2 (OpenAI, hosted API)
2. Sonnet 4.5 (Anthropic, hosted API)
3. Llama 4 Maverick (Meta, self-hosted)

EVALUATION
- Dataset: eval-v1.0 (100 questions, 3 difficulty tiers)
- Conditions: identical retrieval pipeline, tool definitions, and system prompt
- Period: 2026-02-20 to 2026-02-23
- Total evaluation cost: $47 (API fees) + 12 hours (annotation)

ELIMINATIONS
- Llama 4 Maverick: ELIMINATED
  - Citation precision: 0.82 (required ≥0.85)
  - Tool reliability: 0.87 (required ≥0.90)
  - Note: cheapest candidate ($0.02/query) but fails two hard constraints

DECISION
- Selected: GPT-5.2
  - Hard constraints: all pass with comfortable margins
  - Soft weighted total: 0.868 (vs. Sonnet 4.5 at 0.855)
  - Margin: 1.3pp (narrow)

SENSITIVITY
- Weight sensitivity: decision reverses if disambiguation weight > 0.35
  and claim support weight < 0.20. Current product direction does not
  favor this reweighting.
- Cost sensitivity: at >10,000 queries/day, cost difference vs. Sonnet 4.5
  exceeds $6,000/month. Re-evaluate cost weighting at that volume.
- Measurement uncertainty: confidence intervals on citation precision overlap.
  Decision rests on weighted total, not citation precision alone.

FALLBACK
- Sonnet 4.5 is maintained as tested fallback
- Passes all hard constraints independently
- Rollback time: <5 minutes (environment variable switch)

RE-EVALUATION TRIGGERS
- New model release with >5% improvement on citation precision
- GPT-5.2 cost/query exceeding $0.12 (80% of hard constraint)
- GPT-5.2 p95 latency exceeding 4.0s (80% of hard constraint)
- Daily query volume exceeding 5,000 (cost sensitivity threshold)
- Eval-v1.0 refresh (if evaluation set becomes stale due to distribution shift)

RISKS
- Provider lock-in: GPT-5.2 uses OpenAI-specific function calling syntax.
  Mitigation: tool definitions use an abstraction layer (LangChain) that
  supports multiple providers.
- Silent model updates: OpenAI may update GPT-5.2 behavior without notice.
  Mitigation: monthly regression evaluation against eval-v1.0.
- Pricing changes: historical trend is downward, but increases are possible.
  Mitigation: tested fallback on a different provider.
```

The decision record is a living document.
When any re-evaluation trigger fires, the product manager updates the record with the new evaluation results and either confirms the existing decision or initiates a transition plan.
This practice transforms model selection from a one-time event into a continuous governance process.

### 1.8 Cost modeling under uncertainty

Per-query cost, as computed in the baseline sweep, is a necessary but insufficient input to the model selection decision.
A product manager must project costs across the product lifecycle under uncertainty about usage volume, query complexity distribution, and pricing changes. [Sculley et al., 2015]
This section develops a cost model for the RA that addresses these uncertainties and demonstrates how cost analysis feeds back into both model selection and product design.

#### 1.8.1 Decomposing the cost stack

The fully loaded cost of an AI product query is rarely limited to model inference.
A common error is equating "cost per query" with "LLM token cost," ignoring the retrieval, processing, storage, and observability components that may collectively exceed the inference cost for certain query types.

For the RA, the cost stack decomposes as follows:

| Component | Description | Fixed/Variable | RA Estimate (per query) |
|-----------|-------------|----------------|------------------------|
| LLM input tokens | Tokens sent to the model (system prompt + context + query) | Variable | $0.005–0.02 |
| LLM output tokens | Tokens generated by the model (response + tool calls) | Variable | $0.02–0.08 |
| Retrieval API calls | Semantic Scholar queries, arXiv queries | Variable | $0.001–0.005 |
| PDF download | Bandwidth for downloading full-text papers | Variable | $0.001–0.003 |
| PDF parsing | CPU compute for text extraction (PyMuPDF, pdfplumber) | Variable | $0.001–0.005 |
| Vector store operations | Embedding generation + similarity search (when caching is active) | Variable | $0.001–0.003 |
| Logging and observability | Usage tracking, JSONL writes, monitoring | Variable | $0.0005 |
| Infrastructure (amortized) | Server/container, API gateway, storage | Fixed → amortized | $0.005–0.02 |
| **Total per query** | | | **$0.03–0.15** |

The wide range (5× between minimum and maximum) reflects query complexity.
A simple factual question ("Who wrote Attention Is All You Need?") requires one search call, minimal retrieval context, and a short generation—total cost approximately $0.03.
A synthesis question ("What are the main critiques of transformer attention mechanisms in the recent NLP literature, and how have subsequent architectures addressed them?") triggers multiple search-retrieve-read cycles, processes 3–5 full papers, and generates a long structured response—total cost approximately $0.12.

**The importance of per-component tracking.**
A product manager who tracks only aggregate cost per query cannot diagnose cost overruns or optimize the cost structure.
When aggregate cost increases, is it because the LLM is generating longer responses?
Because the retrieval layer is making more API calls?
Because PDF parsing is slower and consuming more compute?
Per-component logging (implemented in the RA via the `UsageLogger` writing to `data/api-usage.jsonl`) enables root-cause analysis of cost anomalies. [Beyer et al., 2016]

#### 1.8.2 Usage distribution modeling

Aggregate cost projections require a model of the expected query distribution.
Observed usage of research tools follows a heavy-tailed distribution: most queries are simple, but a minority of complex queries consume disproportionate resources. [Baeza-Yates & Ribeiro-Neto, 2011]

The RA models its expected query distribution in three tiers:

| Query Tier | Description | Fraction | Avg. Tool Calls | Avg. Input Tokens | Avg. Output Tokens | Est. Cost |
|------------|-------------|----------|-----------------|-------------------|--------------------| ----------|
| Simple (factual) | Single-fact questions with known answers | 60% | 1–2 | 1,500 | 500 | $0.03 |
| Moderate (analytical) | Questions requiring comparison or analysis across papers | 30% | 3–5 | 4,000 | 1,200 | $0.07 |
| Complex (synthesis) | Multi-paper synthesis, critique, or literature review | 10% | 6–10 | 8,000 | 2,500 | $0.12 |
| **Weighted average** | | **100%** | | | | **$0.052** |

This distribution is an assumption that must be validated with production telemetry.
Initial estimates of query distribution are typically wrong by 30–50%. [Jain, 1991]
The most common error is underestimating the proportion of complex queries: users who adopt an AI research tool often shift their behavior toward more ambitious queries than they would attempt manually, because the marginal effort of asking a harder question is low.

**Updating the distribution with production data.**
The cost model should be parameterized so that the distribution can be updated as real usage data accumulates.
After the first month of production, the actual tier fractions can be measured from tool-call counts and token usage logs.
If the actual distribution is 45% simple / 35% moderate / 20% complex (users are more ambitious than expected), the weighted average cost increases from $0.052 to $0.065—a 25% increase that, compounded over thousands of daily queries, materially affects the business model.

#### 1.8.3 Token estimation methodology

Accurate cost projection requires understanding what drives token consumption.
For the RA, token usage is determined by the agent loop structure:

**Input tokens per turn:**
- System prompt: ~800 tokens (fixed)
- User query: ~50–200 tokens (variable)
- Retrieved context (paper abstracts, snippets): ~500–4,000 tokens per retrieval cycle (variable, depends on how many papers are fetched)
- Tool call history (accumulated across the ReAct loop): grows with each iteration

**The compounding cost of multi-turn agents.**
In a ReAct loop, each iteration appends the previous tool call and observation to the context.
By the 5th iteration, the accumulated context may exceed the original query + retrieval context by 3–4×.
This compounding effect means that complex queries are disproportionately expensive—not linearly proportional to the number of tool calls, but super-linearly proportional because each subsequent LLM call processes all previous context.

For the RA, a query requiring 8 tool calls generates approximately:
- Turn 1: 1,500 input tokens
- Turn 2: 2,800 input tokens (turn 1 context + observation)
- Turn 3: 4,200 input tokens
- ...
- Turn 8: ~12,000 input tokens

Total input tokens across all turns: approximately 50,000—far more than the 8,000 "per-query" estimate suggests when counting only the final context window.
This distinction between "context window tokens" and "total API tokens billed" is a common source of cost estimation error in agent-based systems.

**Mitigation strategies for token cost:**
1. **Context pruning:** Summarize previous tool observations instead of including full text. Reduces accumulated context at the cost of some information loss.
2. **Prompt caching:** Both OpenAI and Anthropic offer cached input pricing (typically 10× cheaper than uncached). If the system prompt and common retrieval context are cacheable, input costs decrease substantially.
3. **Early stopping:** If the agent has sufficient information after 3 tool calls, a well-designed prompt can instruct it to synthesize rather than continuing to search. This requires careful prompt engineering to balance thoroughness against cost.

#### 1.8.4 Scaling projections

Given the weighted average cost of $0.052/query, monthly projections under four growth scenarios:

| Scenario | Daily Queries | Monthly Queries | Monthly Cost | Annual Cost |
|----------|---------------|-----------------|--------------|-------------|
| Pilot (alpha) | 100 | 3,000 | $156 | $1,872 |
| Early adoption | 1,000 | 30,000 | $1,560 | $18,720 |
| Growth | 10,000 | 300,000 | $15,600 | $187,200 |
| Scale | 50,000 | 1,500,000 | $78,000 | $936,000 |

These figures assume GPT-5.2 pricing with no optimization.
With prompt caching (reducing input costs by ~80% for cached portions), the growth-stage cost decreases to approximately $9,000–11,000/month—a meaningful reduction that justifies the engineering investment in caching infrastructure at that scale.

**The revenue side.**
Cost projections are meaningless without revenue context.
For the RA, consider two business models:

*Subscription model ($15/month, est. 200 queries/user/month):*
- Cost per user: $0.052 × 200 = $10.40/month
- Gross margin: ($15 − $10.40) / $15 = 30.7%

This margin is dangerously thin.
At scale (with caching), cost per user drops to ~$6–7/month, improving margin to 50–55%.
But during the growth phase, the business operates near breakeven on variable costs alone—before accounting for engineering salaries, infrastructure, and customer acquisition.

*Usage-based model ($0.10/query):*
- Gross margin per query: ($0.10 − $0.052) / $0.10 = 48%

Better margin, but usage-based pricing creates uncertainty for users and may suppress adoption.

A product manager must model both pricing strategies against the cost curve to identify which is viable at each growth stage.
The insight for model selection: **Sonnet 4.5 at $0.05/query (vs. GPT-5.2 at $0.07/query) improves subscription gross margin from 30.7% to 51.3%—a difference that may determine whether the business is fundable.**

#### 1.8.5 The crossover analysis

A critical product decision is the **model hosting crossover point**: the usage volume at which self-hosting becomes cheaper than API access.

For the RA:
- API cost (GPT-5.2): $0.052 × Q per month, where Q is monthly query volume
- Self-hosted cost (Llama 4 Maverick on 4×A100): ~$18,000/month fixed + $0.008 × Q variable

Setting these equal: $0.052Q = $18,000 + $0.008Q → $0.044Q = $18,000 → Q ≈ 409,000 queries/month ≈ 13,600 queries/day.

Below 13,600 queries/day, the API is cheaper.
Above it, self-hosting saves money—**but only if the self-hosted model meets all hard constraints.**
Since Llama 4 Maverick failed the citation precision constraint (Section 1.7.2), the crossover is currently irrelevant.

However, the crossover analysis remains valuable for roadmap planning:
1. **It establishes a volume target.** If the product reaches 13,600 queries/day, self-hosting becomes economically attractive—motivating investment in improving open-weight model quality for the RA's specific use case (e.g., fine-tuning Maverick on citation tasks to close the precision gap).
2. **It quantifies the value of model improvement.** If Maverick's citation precision could be improved from 0.82 to 0.87 through fine-tuning (a topic addressed in Chapter 2), the self-hosting option becomes viable and saves $30,000+/month at scale.
3. **It creates a contingency plan.** If API pricing increases unexpectedly, the crossover point drops—and having a self-hosting plan ready reduces the urgency of the pricing shock.

**Hidden costs of self-hosting.**
The crossover analysis above considers only compute costs.
Self-hosting introduces additional costs that are frequently underestimated: [Patterson et al., 2021]

- **Engineering time:** Setting up and maintaining a model serving stack (vLLM, TGI, or similar) requires specialized ML engineering. Estimate 0.5–1.0 FTE ongoing.
- **On-call burden:** Self-hosted models require incident response for GPU failures, OOM errors, inference hangs. This is a 24/7 responsibility that API providers absorb.
- **Scaling complexity:** Auto-scaling GPU instances is harder than auto-scaling API calls. Over-provisioning wastes money; under-provisioning causes latency spikes.
- **Model updates:** When Meta releases Llama 4.1, the self-hosted deployment requires testing, validation, and rollout—effort that API providers handle transparently.

A more complete crossover analysis adds $5,000–15,000/month for these hidden costs, pushing the breakeven point to approximately 20,000+ queries/day.

#### 1.8.6 Pricing risk and contractual exposure

Model providers change pricing.
Between 2023 and 2026, OpenAI reduced GPT-4-class pricing by approximately 90%, while simultaneously deprecating older models and introducing new pricing tiers for cached vs. uncached input. [OpenAI, 2024]

A product manager must account for four categories of pricing risk:

1. **Price decreases** — favorable for the product, but also benefit competitors. A price decrease that makes the product more profitable also lowers the barrier for new entrants using the same model.

2. **Price increases** — rare in the historical trend but possible, especially for specialized or high-demand models. Mitigation: maintain a tested fallback model on a different provider at all times. The RA's fallback (Sonnet 4.5 on Anthropic) ensures that no single provider has pricing leverage over the product.

3. **Model deprecation** — the provider retires the model entirely, requiring migration to a successor. OpenAI deprecated GPT-4 Turbo, GPT-3.5 Turbo, and several other models between 2024–2026. Mitigation: the decision record (Section 1.7.4) identifies re-evaluation triggers; the baseline sweep procedure (Section 1.4) can be re-run on the replacement model within days if the evaluation infrastructure is maintained. **The evaluation infrastructure is the hedge against deprecation.**

4. **Rate limit changes** — the provider restricts throughput, effectively increasing the cost of scaling. Mitigation: implement client-side rate limiting, queue management, and request prioritization; maintain capacity on a secondary provider for overflow.

**Contractual considerations.**
At scale (>$10,000/month in API spend), direct contracts with model providers typically offer volume discounts (10–30% below list pricing), committed throughput guarantees, and advance notice of deprecation.
A product manager should initiate contract discussions before reaching scale, not after—negotiating from a position of projected volume rather than current spend.

### 1.9 Production readiness gate

Model selection produces a candidate; production readiness determines whether that candidate can be deployed to users.
The gap between "works in evaluation" and "works in production" is the dominant source of AI product failure—not because teams select the wrong model, but because they deploy the right model without the operational infrastructure required to sustain it. [Sculley et al., 2015]

This section defines the production readiness gate for the RA: the set of conditions that must be satisfied before the selected model can serve real users.

#### 1.9.1 Why evaluation performance is insufficient

A model that achieves 0.91 citation precision on eval-v1.0 will not achieve 0.91 citation precision in production.
Several systematic factors degrade production performance relative to evaluation:

**Distribution shift.** The evaluation set represents the product manager's best guess at the query distribution, constructed before real users interact with the system. Real users will ask questions that differ from the evaluation set in ways that are difficult to predict. They will ask about topics not covered by the evaluation set, use phrasing that differs from the evaluation prompts, and combine the RA with workflows the designers did not anticipate.

For the RA, the evaluation set was constructed by domain experts who formulated well-structured research questions.
Real users—especially those new to a research domain—may ask vague, poorly scoped, or ambiguous questions ("tell me about AI safety" vs. "What are the main technical approaches to alignment in large language models published since 2023?").
The model's citation precision on vague queries is likely lower than on well-structured ones, because vague queries require more disambiguation and the retrieval results are noisier.

**Adversarial and edge-case inputs.** Evaluation sets typically exclude deliberately adversarial inputs (unless specifically designed for robustness testing). In production, users will inevitably test the system's boundaries—asking about topics outside the academic domain, requesting actions the system is not designed for, or providing inputs that trigger unexpected model behavior.

**Infrastructure variance.** Evaluation is typically conducted under controlled conditions: low concurrency, stable network, no competing workload. Production introduces variable API latency (due to provider-side load), concurrent requests (causing queuing), network interruptions, and infrastructure failures.

**Temporal drift.** The evaluation set is a snapshot; the world changes. New papers are published, terminology evolves, and the distribution of user queries shifts over time. A model that performs well on eval-v1.0 in February 2026 may degrade by August 2026 if the evaluation set is not refreshed.

These factors collectively motivate the production readiness gate: a set of operational requirements that must be met *in addition to* evaluation performance before the model is deployed.

#### 1.9.2 Availability and compound reliability

The system must respond to queries during stated operating hours.
For the RA, the availability target is 99.5% (approximately 3.6 hours of permissible downtime per month).

This target appears modest, but it constrains architectural choices more than it may seem.
The RA depends on multiple external services, each with its own reliability:

| Dependency | Estimated Availability | Failure Mode |
|------------|----------------------|--------------|
| OpenAI GPT-5.2 API | 99.9% | Request errors, rate limiting, model degradation |
| Semantic Scholar API | 99.5% | Downtime, rate limiting, stale index |
| arXiv API | 99.0% | Maintenance windows, XML parsing errors |
| Internal infrastructure | 99.9% | Server crashes, deployment errors |

The compound availability of sequential dependencies (where *all* must succeed for a query to complete) is the product of individual availabilities: [Beyer et al., 2016]

99.9% × 99.5% × 99.0% × 99.9% ≈ 98.3%

This is **below the 99.5% target** by a significant margin—equivalent to approximately 12.5 hours of downtime per month instead of the budgeted 3.6 hours.

**Mitigation strategies:**

1. **Graceful degradation.** Not every dependency is required for every query. If the arXiv API is unavailable, the RA can still search Semantic Scholar and return results based on metadata and abstracts—a degraded but functional response. The system should detect which dependencies are available and adapt its behavior accordingly, rather than failing entirely when any single dependency is down.

2. **Caching.** Frequently accessed papers (high-citation papers in popular fields) can be cached locally, reducing dependency on external APIs for common queries. Cache hit rates of 20–40% on retrieval API calls can meaningfully improve compound availability.

3. **Redundant retrieval sources.** Adding a third retrieval source (e.g., CrossRef, OpenAlex) provides redundancy: if Semantic Scholar is down, CrossRef can serve metadata queries. This increases integration complexity but improves the reliability of the retrieval layer.

4. **Timeout and fallback.** If an external API does not respond within a defined timeout (e.g., 3 seconds), the system falls back to cached results or responds with a partial answer plus an explicit disclaimer ("Some sources may not be available; results are based on cached data").

With graceful degradation and caching, the effective availability of the RA's core functionality (returning a cited answer, even if not using all retrieval sources) can exceed 99.5% even when individual dependencies fall below their SLAs.

#### 1.9.3 Latency under load

Evaluation measures latency on isolated queries—one request at a time, with no queuing.
Production latency includes queuing time when concurrent users exceed the system's throughput capacity.

**Estimating concurrency requirements.**
Little's Law relates mean concurrency (L), arrival rate (λ), and mean service time (W): L = λW. [Jain, 1991]

For the RA:
- Mean service time (W): 3.0 seconds (observed in evaluation)
- Peak arrival rate (λ): estimated at 10 queries/second during peak hours (based on 10,000 daily queries concentrated in a 6-hour active window)
- Mean concurrency: L = 10 × 3.0 = 30 concurrent requests

This means the system must support 30 concurrent requests without significant queuing delay.
The constraint propagates to the LLM API: OpenAI's rate limits for GPT-5.2 must accommodate 30 concurrent requests, and the retrieval APIs must similarly support the throughput.

**Queuing behavior.**
When arrival rate approaches or exceeds service capacity, queuing delay grows non-linearly.
For an M/M/1 queue (a simple model), mean response time is W / (1 − ρ), where ρ = λW/capacity is the utilization factor. [Jain, 1991]
At 80% utilization, mean response time is 5× the service time; at 90% utilization, it is 10×.

This non-linearity means that a system operating comfortably at 70% utilization can violate the latency constraint during traffic spikes that push utilization above 85%.
Mitigation: provision capacity to maintain utilization below 70% during expected peaks, or implement request queuing with explicit timeouts and user-facing wait indicators.

**Tail latency.**
The p95 latency constraint (≤5.0s) means that 95% of queries must complete within 5 seconds, but 5% may take longer.
The distribution of latency in agent-based systems is heavy-tailed: most queries complete in 2–3 seconds, but complex queries requiring 6+ tool calls can take 10–15 seconds.
A product manager must decide how to handle the tail:

- **Accept the tail:** Display a loading indicator and let long queries complete. Acceptable if the user has context (e.g., "Searching 5 papers...").
- **Timeout and return partial results:** After 5 seconds, return whatever the agent has synthesized so far, with a disclaimer that the search is incomplete. This satisfies the latency constraint but may reduce answer quality.
- **Offer asynchronous mode:** For complex queries, offer to email or notify the user when the analysis is complete. This reframes the latency constraint as a UX design decision rather than a hard technical constraint.

The RA implements a combination: simple queries return synchronously within the latency budget; complex queries that exceed 5 seconds display intermediate results ("Found 3 relevant papers so far...") and continue processing.

#### 1.9.4 Data freshness

Academic papers are published continuously.
The RA's retrieval sources (Semantic Scholar, arXiv) index new papers with varying latency: arXiv within hours of submission, Semantic Scholar within days of publication (due to metadata enrichment and citation graph updates).

A production system must define a **freshness SLO**: the maximum acceptable delay between a paper's publication and its availability in the RA's retrieval results.

For the RA, the freshness SLO is defined as:
- arXiv preprints: available within 48 hours of posting
- Published papers (with DOI): available within 7 days of Semantic Scholar indexing

This SLO is achievable with the current retrieval architecture (which queries external APIs in real-time) but would be violated if the system migrated to a fully cached/indexed architecture without periodic refresh.

**Why freshness matters for product trust.**
If a researcher asks about a paper published yesterday and the RA cannot find it, the researcher's trust in the system decreases disproportionately—even if the system correctly handles 99% of queries about older papers.
Freshness failures are highly salient because the user *knows* the paper exists (they may have just seen it on arXiv) and interprets the RA's failure as incompetence rather than latency.

This is an instance of a general principle: **user trust is degraded more by failures on known-answer queries than by failures on ambiguous queries.** A product manager must identify these high-salience failure modes and ensure the system handles them reliably, even if they represent a small fraction of total queries.

#### 1.9.5 Error budgets

An error budget quantifies the acceptable amount of unreliability over a given period. [Beyer et al., 2016]
It operationalizes the relationship between reliability investment and feature velocity: as long as the error budget is not exhausted, the team can ship changes (new features, prompt updates, model upgrades); when it is exhausted, reliability work takes priority.

For the RA, error budgets are defined per hard constraint:

| Constraint | Target | Error Budget (per month) | Monitoring Method |
|------------|--------|--------------------------|-------------------|
| Availability | 99.5% | 3.6 hours downtime | Uptime monitoring (ping + synthetic queries) |
| Citation precision | ≥0.85 | ≤15% of sampled queries may have incorrect citations | Daily automated eval on 50-query sample |
| p95 latency | ≤5.0s | ≤5% of queries may exceed 5s | Latency percentile dashboard |
| Tool reliability | ≥0.90 | ≤10% of queries may have all tool calls fail | Tool-call success rate counter |

**The error budget as a governance mechanism.**
Error budgets are more than monitoring thresholds—they are a governance mechanism that aligns engineering priorities with product requirements.

Consider a scenario: the ML engineer wants to deploy an updated system prompt that improves answer completeness (a soft constraint) but has not been fully evaluated for citation precision impact.
Without an error budget, this becomes a judgment call with no clear decision framework.
With an error budget, the decision is structured:

1. Check current citation precision error budget consumption: 8% of monthly budget used.
2. Deploy the new prompt to 10% of traffic (canary).
3. Monitor citation precision on the canary population for 24 hours.
4. If the canary consumes error budget at an acceptable rate (projected to stay under 50% of monthly budget), expand to 100%.
5. If the canary shows degradation (projected to exceed monthly budget), roll back.

This procedure allows the team to ship improvements quickly while maintaining a safety net.
It replaces the false binary of "ship everything" vs. "test everything exhaustively" with a calibrated approach that matches the level of caution to the remaining error budget. [Beyer et al., 2016]

#### 1.9.6 Monitoring and observability

Production AI systems require monitoring at three levels: [Breck et al., 2017]

**Level 1: Infrastructure monitoring.**
Standard application monitoring: API latency distributions, error rates by endpoint, throughput (queries/second), CPU/memory utilization, cost accumulation rate.
This is table-stakes for any production system and is well-supported by existing monitoring tools (Prometheus, Datadog, CloudWatch, etc.).

For the RA, infrastructure monitoring is implemented through the `UsageLogger`, which writes a JSONL entry for every API call with timestamp, endpoint, response time, token counts, and cost estimate.
Aggregation queries over this log produce the dashboards needed for infrastructure monitoring.

**Level 2: Model behavior monitoring.**
This is specific to AI products and is frequently neglected.
Model behavior monitoring detects degradation in output quality that does not manifest as infrastructure errors.

For the RA, model behavior monitoring includes:
- **Daily automated evaluation:** A pipeline that runs 50 randomly sampled production queries through the evaluation scorer (automated citation precision check, claim-support check). The automated scorer is calibrated against human judgments (agreement rate ≥0.85) to ensure that automated monitoring is a reliable proxy for human quality assessment.
- **Output distribution monitoring:** Track the distribution of output characteristics: average response length, number of citations per response, fraction of responses with zero citations, fraction of responses that include "I could not find" disclaimers. Sudden shifts in these distributions may indicate model degradation even if the automated quality score has not yet detected it.
- **Tool-call pattern monitoring:** Track the average number of tool calls per query, the distribution of tool types used, and the fraction of queries where the agent "gives up" (reaches max iterations without a satisfactory answer). An increase in max-iteration hits may indicate that the model's reasoning ability has degraded (e.g., due to a silent provider update).

**Level 3: Data distribution monitoring.**
Detecting shifts in the input distribution that may degrade model performance even if the model itself has not changed.

For the RA:
- **Query topic distribution:** Cluster production queries by topic (using embeddings or keyword analysis) and compare against the evaluation set's topic distribution. If a significant fraction of production queries falls in topics not represented by the evaluation set, the automated quality estimates may be unreliable.
- **Query complexity distribution:** Track the fraction of queries in each complexity tier (simple/moderate/complex). If users shift toward more complex queries (which is likely as they develop trust in the system), the effective cost per query and error rates may increase.
- **Retrieval result quality:** Monitor the average number of results returned per search, the fraction of searches with zero results, and the average citation count of retrieved papers. A decrease in retrieval quality may indicate changes in external API behavior or indexing coverage.

#### 1.9.7 Rollback and incident response

A production readiness gate must include a rollback plan: what happens when the deployed model produces unacceptable results? [Beyer et al., 2016]

**The rollback hierarchy for the RA:**

1. **Prompt rollback (minutes).** If a prompt change causes degradation, revert to the previous prompt version. This is the fastest rollback and requires only a configuration change. The RA maintains version-controlled prompts in the codebase; reverting is a git revert + deployment.

2. **Model rollback (minutes).** Switch from GPT-5.2 to Sonnet 4.5 (the pre-tested fallback) via environment variable change. No code deployment required. Rollback time: <5 minutes. This is the primary rollback mechanism for model-level failures.

3. **Feature rollback (minutes).** Disable specific agent capabilities (e.g., full-text PDF parsing, citation chasing) if a particular tool or pipeline stage is causing failures, while maintaining core search-and-cite functionality. Implemented via feature flags.

4. **Full rollback (hours).** Revert the entire system to a known-good state (previous deployment). This is the nuclear option, used only when multiple components are simultaneously failing and the root cause is unclear.

**Testing the rollback plan.**
A rollback plan that has never been tested is not a plan—it is a hope. [Beyer et al., 2016]
The RA's rollback plan is tested quarterly:
- The fallback model (Sonnet 4.5) is run against the current evaluation set to confirm it still meets hard constraints. Model providers occasionally change model behavior through silent updates; a fallback that passed constraints six months ago may not pass today.
- The prompt rollback procedure is exercised: a previous prompt version is deployed to a staging environment and validated.
- The feature flag system is tested: each flag is toggled off individually and the system's degraded behavior is verified.

#### 1.9.8 The silent model update problem

A unique challenge of AI products that rely on hosted API models is the **silent model update**: the provider modifies the model's behavior without explicit notification. [Sculley et al., 2015]

This is not hypothetical.
Between 2023 and 2026, multiple instances were documented where hosted model behavior changed between API calls on the same model version, causing unexpected regressions in downstream applications.
The changes are typically minor (adjustments to safety filters, output formatting, or sampling parameters) but can be consequential for products with tight quality constraints.

For the RA, the mitigation strategy is:
1. **Continuous evaluation:** The daily automated evaluation (Level 2 monitoring) detects behavioral changes within 24 hours. If citation precision drops below the alert threshold, investigation begins immediately.
2. **Versioned snapshots:** When available, use pinned model versions (e.g., `gpt-5.2-2026-02-15`) rather than the latest alias. This prevents silent updates but requires periodic manual upgrades.
3. **Provider communication:** At sufficient spend levels, establish a direct relationship with the provider's developer relations team to receive advance notice of planned changes.
4. **Evaluation-as-contract:** The evaluation set serves as a de facto contract with the model provider: "Our product depends on this model achieving X on these inputs. If an update degrades performance below X, we will roll back." This framing makes evaluation infrastructure an asset, not a cost.

#### 1.9.9 The readiness checklist

Before a model selection can be considered production-ready, the following gates must be satisfied.
This checklist is intentionally specific to encourage rigorous verification rather than box-checking.

**Evaluation gates:**
- [ ] All hard constraints pass on the selected model (eval-v1.0 or later)
- [ ] Confidence intervals on hard constraint metrics are documented
- [ ] A tested fallback model exists and passes all hard constraints independently
- [ ] The evaluation set is versioned and the version is recorded in the decision record

**Operational gates:**
- [ ] Compound availability has been calculated and meets the target (with mitigations)
- [ ] Latency under expected peak load has been estimated (queuing model or load test)
- [ ] Data freshness SLO is defined and achievable with current architecture
- [ ] Error budgets are defined per hard constraint with documented alert thresholds

**Monitoring gates:**
- [ ] Infrastructure monitoring is active (latency, error rates, cost, throughput)
- [ ] Model behavior monitoring is active (daily automated eval, output distribution tracking)
- [ ] Data distribution monitoring is active (query topic/complexity tracking)
- [ ] Alerting thresholds are configured and routed to the on-call team

**Incident response gates:**
- [ ] Rollback procedure is documented for each level (prompt, model, feature, full)
- [ ] Rollback has been tested within the last quarter
- [ ] The fallback model has been re-validated within the last quarter
- [ ] On-call responsibilities are assigned and documented

**Business gates:**
- [ ] Cost projections exist for current volume and 3 growth scenarios
- [ ] Pricing risk is mitigated (fallback provider, contractual terms reviewed)
- [ ] The decision record is complete and reviewed by at least one stakeholder
- [ ] Re-evaluation triggers are defined and monitored

### 1.10 Chapter summary and checklist

This chapter established a procedure for translating product requirements into a defensible model selection decision.
The key contribution is the separation of the problem into sequential stages—constraint specification, landscape mapping, baseline sweep, scoring, sensitivity analysis, cost modeling, and production readiness—each of which produces an auditable artifact.

The procedure is designed to be **reusable across AI products**, not specific to the RA.
The RA was used throughout as an instantiation of the general procedure, demonstrating how abstract principles translate into concrete decisions.
A product manager building a different AI product (a customer support agent, a code generation tool, a medical Q&A system) would follow the same stages with different constraints, different candidate models, and different production requirements—but the reasoning structure is identical.

**Key principles established in this chapter:**

1. **Model selection is a constraint satisfaction problem, not an optimization problem.** The first task is to eliminate candidates that violate hard constraints, not to find the "best" model on aggregate metrics. Mixing hard and soft constraints in a single scoring function can mask disqualifying failures.

2. **Requirements must be operationalized before they are useful.** "Accurate" is not a requirement; "citation precision ≥0.85 on eval-v1.0" is a requirement. Operationalization forces clarity about what is being measured, what passes, and how measurement is conducted.

3. **Fair comparison requires controlled conditions.** Models must be evaluated under identical system configurations (same retrieval pipeline, same tools, same prompts) to isolate model capability from system design. Uncontrolled comparisons measure engineering effort, not model quality.

4. **Sensitivity analysis is not optional.** A decision that reverses under plausible perturbations to weights, thresholds, or cost assumptions is not robust. Documenting sensitivity conditions enables principled re-evaluation when circumstances change.

5. **Cost modeling must extend beyond per-query estimates.** Usage distribution, scaling projections, hosting crossover analysis, and pricing risk assessment are required to ensure the model selection remains viable as the product grows.

6. **Evaluation performance does not predict production performance.** Distribution shift, adversarial inputs, infrastructure variance, and temporal drift systematically degrade production quality. The production readiness gate addresses these factors through availability engineering, monitoring, error budgets, and incident response.

7. **The decision record is the primary artifact.** The output of model selection is not a model name but a documented reasoning chain that enables auditability, reversibility, and institutional memory.

**The complete model selection procedure:**

The following checklist synthesizes the full procedure into a working document.
It is intended to be used as a living checklist during the model selection process, updated as each stage is completed, and preserved as part of the decision record.

**Stage 1: Constraint specification**
- [ ] Product requirements are enumerated from stakeholder interviews and user research
- [ ] Each requirement is classified as hard (violation → elimination) or soft (optimize)
- [ ] Each requirement is operationalized with: metric, threshold, population, measurement procedure
- [ ] Thresholds include documented rationale (why this specific number?)
- [ ] Hard constraints are validated: would the product be unshippable if this constraint is violated?
- [ ] Soft constraint weights are assigned using a structured method (e.g., swing weighting)
- [ ] The constraint set is reviewed and approved by at least one stakeholder outside the ML team

**Stage 2: Landscape mapping**
- [ ] Hosted API and self-hosted (open-weight) candidates are enumerated
- [ ] Candidates are filtered by obvious disqualifiers: context length, modality support, regional availability, data governance requirements
- [ ] A shortlist of 3–5 candidates is established for evaluation
- [ ] For self-hosted candidates, infrastructure requirements and costs are estimated

**Stage 3: Baseline sweep**
- [ ] A representative evaluation set exists (≥50 prompts, ideally 100+, across difficulty tiers)
- [ ] The evaluation set is versioned (e.g., eval-v1.0)
- [ ] Metrics aligned with constraints are implemented and automated where possible
- [ ] Each candidate is evaluated under identical, controlled conditions
- [ ] Results are recorded with confidence intervals (sample size permitting)
- [ ] The total cost of the evaluation (API fees, annotation time) is documented

**Stage 4: Decision**
- [ ] Hard constraints are applied as elimination gates (binary pass/fail)
- [ ] Eliminated candidates are documented with specific constraint violations
- [ ] Soft constraints are scored and weighted for surviving candidates
- [ ] The weighted total is computed and the leading candidate is identified
- [ ] Sensitivity analysis is performed on: weights, thresholds, cost, measurement uncertainty
- [ ] Conditions that would reverse the decision are documented explicitly
- [ ] A decision record is drafted with: context, candidates, evaluation, eliminations, decision, sensitivity, fallback, re-evaluation triggers, risks

**Stage 5: Cost modeling**
- [ ] Per-query cost is decomposed into components (inference, retrieval, parsing, storage, overhead)
- [ ] Usage distribution is modeled across complexity tiers (with acknowledged uncertainty)
- [ ] Monthly costs are projected under 3+ growth scenarios
- [ ] Revenue model is compared against cost projections (gross margin analysis)
- [ ] The self-hosting crossover point is calculated (if applicable)
- [ ] Pricing risk is assessed: price increase, deprecation, rate limit changes
- [ ] Mitigation is in place: tested fallback on a different provider

**Stage 6: Production readiness**
- [ ] Compound availability is calculated and meets the target (with mitigations documented)
- [ ] Latency under expected peak load is estimated (queuing model, Little's Law, or load test)
- [ ] Data freshness SLO is defined and achievable
- [ ] Error budgets are defined per hard constraint with alert thresholds
- [ ] Monitoring covers all three levels: infrastructure, model behavior, data distribution
- [ ] Alerting is configured and routed to responsible parties
- [ ] Rollback procedures are documented for each level (prompt, model, feature, full)
- [ ] Rollback has been tested (not just documented)
- [ ] The fallback model has been re-validated recently
- [ ] The decision record is finalized and archived

**What this chapter does not cover.**
This chapter addresses model *selection*—choosing and validating a model for an AI product.
It does not address model *customization* (fine-tuning, which is the subject of Chapter 2), *evaluation infrastructure at scale* (which is the subject of Chapter 3), or *system architecture design* (which is the subject of Chapter 4).
The model selection procedure produces a candidate and a production readiness plan; the subsequent chapters address how to improve that candidate, how to maintain quality over time, and how to build the system around it.

### References

- Baeza-Yates, R., & Ribeiro-Neto, B. (2011). *Modern Information Retrieval: The Concepts and Technology Behind Search* (2nd ed.). Addison-Wesley.
- Beyer, B., Jones, C., Petoff, J., & Murphy, N. R. (Eds.). (2016). *Site Reliability Engineering: How Google Runs Production Systems*. O'Reilly Media.
- Bornmann, L., & Mutz, R. (2015). Growth Rates of Modern Science: A Bibliometric Analysis Based on the Number of Publications and Cited References. *Journal of the Association for Information Science and Technology*, 66(11).
- Breck, E., Cai, S., Nielsen, E., Salib, M., & Sculley, D. (2017). The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction. *IEEE International Conference on Big Data*.
- Card, S. K., Robertson, G. G., & Mackinlay, J. D. (1991). The Information Visualizer: An Information Workspace. *ACM CHI*.
- Clark, K., et al. (2020). ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators. *ICLR*.
- Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *NAACL*.
- Jain, R. (1991). *The Art of Computer Systems Performance Analysis: Techniques for Experimental Design, Measurement, Simulation, and Modeling*. Wiley.
- Keeney, R. L., & Raiffa, H. (1993). *Decisions with Multiple Objectives: Preferences and Value Tradeoffs*. Cambridge University Press.
- Lewis, P., Perez, E., Piktus, A., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS*.
- Liu, Y., et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach. *arXiv*.
- Marcus, G., & Davis, E. (2019). *Rebooting AI: Building Artificial Intelligence We Can Trust*. Pantheon.
- Nygard, K. E., & Kramer, N. (1988). Decision Tables in Software Engineering. *Journal of Systems and Software*, 8(4).
- OpenAI. (2024). Pricing and Model Deprecation Updates. *OpenAI Platform Documentation*.
- Patterson, D., Gonzalez, J., Le, Q., et al. (2021). Carbon Emissions and Large Neural Network Training. *arXiv:2104.10350*.
- Rajpurkar, P., Jia, R., & Liang, P. (2018). Know What You Don't Know: Unanswerable Questions for SQuAD. *ACL*.
- Saltelli, A., Ratto, M., Andres, T., et al. (2008). *Global Sensitivity Analysis: The Primer*. Wiley.
- Sculley, D., Holt, G., Golovin, D., et al. (2015). Hidden Technical Debt in Machine Learning Systems. *NeurIPS*.
- Vaswani, A., et al. (2017). Attention is All You Need. *NeurIPS*.

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
