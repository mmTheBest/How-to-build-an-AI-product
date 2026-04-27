# Chapter 2: Context Engineering and Prompt Design

In the development of an AI product, the system prompt is the first and most consequential design artifact after model selection.
It is the mechanism through which product requirements, behavioral constraints, and output specifications are communicated to the model at inference time.
Unlike traditional software, where behavior is determined by compiled code, the behavior of an LLM-based product is determined in large part by natural-language instructions that are interpreted probabilistically.

This chapter examines the principles and practice of context engineering for AI products, using the academic research assistant (Arxie) as the running example.
The term "context engineering" is preferred over "prompt engineering" because the relevant design space extends beyond the system prompt to encompass tool descriptions, retrieval context, output post-processing, and the allocation of the context window across competing demands.

## 2.1 The system prompt as product specification

The system prompt of an AI product is functionally equivalent to a product specification: it defines what the system does, how it behaves, and what constraints it observes.
Every product requirement that the model must satisfy at inference time—behavioral rules, output format, tool-use policies, tone, safety constraints—must be encoded in the system prompt or in the tool descriptions that accompany it.

This equivalence has a profound implication: **the quality of the system prompt places a ceiling on the quality of the product.**
A model that is capable of producing correct, well-cited research summaries will fail to do so if the system prompt does not instruct it to cite sources, specify the citation format, or define what "correct" means in context. [Reynolds & McDonell, 2021]

### 2.1.1 From product requirements to prompt instructions

The translation from product requirements to prompt instructions is not a creative exercise—it is a systematic mapping.
Each product requirement identified in the constraint specification (Chapter 1) must be traced to one or more prompt instructions that operationalize it.

For Arxie, the mapping is as follows:

| Product Requirement | Constraint (Ch. 1) | Prompt Instruction |
|--------------------|--------------------|-------------------|
| Citations must be accurate | Citation precision ≥0.85 | "Always cite papers using (Author et al., Year) format inline. Every non-trivial factual claim should be backed by at least one citation." |
| Answers must be grounded in evidence | Claim support ≥0.80 | "Do not answer from prior knowledge alone; ground answers in retrieved paper metadata." |
| The system must use retrieval tools | Tool reliability ≥0.90 | "You MUST call search_papers at least once before finalizing any answer." |
| Graceful handling of unknown topics | Trust design | "If you cannot find relevant papers, say so explicitly rather than guessing." |
| Structured output | Output structure ≥0.90 | "Provide a References section at the end listing all cited papers." |
| Full-text grounding for specific claims | Claim support (deep mode) | "When a user asks about specific methods, results, experiments, discussion points, or conclusions from a paper, call read_paper_fulltext for that paper before answering." |

This mapping serves two purposes.
First, it ensures **completeness**: every product requirement has a corresponding prompt instruction.
A requirement without a prompt instruction is a requirement the model cannot satisfy, because the model has no other channel through which to receive the instruction at inference time.
Second, it enables **traceability**: when the product fails to meet a constraint, the product manager can trace the failure back to a specific prompt instruction (or the absence of one) and determine whether the fix is a prompt change, a tool change, or a model change.

### 2.1.2 The anatomy of Arxie's system prompt

Arxie's system prompt, as deployed in v1.0, is structured in four sections.
This structure is not arbitrary—it reflects a deliberate decomposition of the prompt into components with different purposes and different rates of change.

**Section 1: Role and identity (~50 tokens).**
```
You are an Academic Research Assistant.
You have access to tools for academic literature retrieval.
```

This section establishes the model's role and primes it for the domain.
Research on role prompting indicates that explicit role assignment improves task performance on domain-specific tasks, likely because it activates relevant learned associations in the model's weights. [Zheng et al., 2024]
The role section is the most stable part of the prompt—it changes only if the product's fundamental identity changes.

**Section 2: Behavioral goals (~100 tokens).**
```
Your goals:
- Search for relevant papers and gather evidence from credible sources.
- Synthesize findings into a clear, structured answer.
- Always cite papers using (Author et al., Year) format inline.
- Every non-trivial factual claim should be backed by at least one citation.
- Provide a References section at the end listing all cited papers.
- Do not answer from prior knowledge alone; ground answers in retrieved paper metadata.
```

This section translates the soft constraints from Section 1.7 into behavioral instructions.
Each instruction is positive ("do X") rather than negative ("don't do Y") where possible, because LLMs are more reliable at following affirmative instructions than prohibitions—a finding consistent with instruction-following research. [Ouyang et al., 2022]

The instruction "Do not answer from prior knowledge alone" is a notable exception: it is phrased as a prohibition because the positive formulation ("always use retrieval tools") was found during development to be insufficient.
The model would sometimes retrieve papers, then generate an answer that drew primarily on its parametric knowledge rather than the retrieved content, occasionally contradicting the retrieved evidence.
The explicit prohibition reduced this behavior from approximately 15% of responses to approximately 5%.

**Section 3: Tool-use policy (~80 tokens).**
```
Tool-use rules:
- You MUST call search_papers at least once before finalizing any answer.
- Use search_papers first, then get_paper_details for promising results.
- When a user asks about specific methods, results, experiments, discussion
  points, or conclusions from a paper, call read_paper_fulltext for that
  paper before answering.
```

This section is the most operationally consequential.
It defines the agent's workflow—the sequence of actions the model should take to fulfill a query.
The instruction "You MUST call search_papers at least once" is a hard behavioral gate that directly supports the tool reliability constraint (≥0.90).

During development, Arxie exhibited a failure mode where the model would sometimes skip tool calls entirely and generate an answer from parametric knowledge, producing responses that appeared well-cited but contained fabricated references.
The "MUST" instruction, capitalized for emphasis, reduced tool-call skip rate from approximately 12% to under 3%.
This is an instance of a general pattern: **the system prompt must encode not just the desired output, but the desired process.**
For agent-based products, process compliance (using the right tools in the right order) is often more important than output quality on any single dimension, because process failures (skipping retrieval) cascade into output failures (hallucinated citations).

**Section 4: Output constraints (~60 tokens).**
These instructions specify the format and structure of the response.
In Arxie's case, the output constraints require inline citations in (Author et al., Year) format and a References section at the end.

The output constraint section is the most frequently updated part of the prompt.
During Arxie's development, the citation format instruction was revised five times:
1. "Cite your sources." → Model cited inconsistently (sometimes footnotes, sometimes inline, sometimes no citations).
2. "Cite sources using APA format." → Model used various APA-like formats inconsistently.
3. "Cite sources using (Author, Year) format inline." → Model sometimes omitted "et al." for multi-author papers.
4. "Cite sources using (Author et al., Year) format inline for papers with 3+ authors." → Consistent formatting achieved.
5. "Always cite papers using (Author et al., Year) format inline. Every non-trivial factual claim should be backed by at least one citation." → Added the "every claim" instruction to increase citation density.

Each revision was motivated by a specific failure observed during evaluation.
This iterative refinement process—observe failure, diagnose cause, revise instruction, re-evaluate—is the standard development loop for prompt engineering.
It is empirical and incremental, not theoretical. [Zamfirescu-Pereira et al., 2023]

### 2.1.3 Prompt instructions that failed

Not every product requirement can be enforced through prompt instructions.
Documenting failed instructions is as valuable as documenting successful ones, because it identifies the boundary of what prompt engineering can achieve—a boundary that determines when system-level solutions or later model customization are required (see Chapters 4 and 6).

**Failed instruction: "Do not cite papers that do not exist."**
This instruction was added to address the hallucinated citation problem (the model generates plausible-looking references that do not correspond to real papers).
The instruction had no measurable effect on hallucination rate.
The reason is straightforward: the model cannot distinguish between real and fabricated citations from the instruction alone—it would need access to a verification tool or database.
The solution was a **system-level fix**, not a prompt fix: a post-processing verification step that checks each cited paper against the Semantic Scholar API and removes unverifiable citations.

**Failed instruction: "Limit your response to 500 words."**
This instruction was intended to control response length for the standard query mode.
In practice, the model frequently exceeded the limit (by 20–50%) or truncated responses awkwardly mid-sentence to comply.
Length control via prompt instructions is unreliable because the model generates tokens sequentially and cannot accurately predict the total length of its output during generation.
The solution was a **system-level fix**: implementing output truncation with a sentence-boundary-aware cutoff in post-processing.

**Failed instruction: "If the query is ambiguous, ask a clarifying question before searching."**
This instruction was intended to improve disambiguation (soft constraint, target ≥0.85).
In practice, the model over-applied it: approximately 30% of non-ambiguous queries triggered unnecessary clarification questions, degrading user experience.
The instruction was removed and disambiguation was handled through the retrieval strategy instead: the agent searches with the query as-given, evaluates the relevance of results, and only asks for clarification if search results are irrelevant or contradictory.

These failures illustrate a general principle: **prompt instructions are effective for specifying output format and behavioral policies, but unreliable for controlling processes that require judgment, verification, or precise quantitative constraints.**
When a product requirement falls in the latter category, it must be implemented at the system level—through tools, post-processing, or architectural design.

### 2.1.4 The prompt as a versioned artifact

The system prompt is code.
It determines the product's behavior as directly as any function or class definition.
It follows that the system prompt should be managed with the same rigor as code: version-controlled, reviewed, tested, and deployed through a defined process. [Zamfirescu-Pereira et al., 2023]

For Arxie, the system prompt is stored in the codebase (`src/ra/agents/research_agent.py`) and changes to it are committed with descriptive messages, reviewed for constraint alignment, and evaluated against the eval-v1.0 suite before deployment.

This practice enables:
1. **Rollback.** If a prompt change degrades performance, the previous version can be restored immediately (git revert). This is the fastest rollback mechanism available (Section 1.9.7).
2. **A/B testing.** Different prompt versions can be deployed to different user segments to measure the impact of specific instruction changes on product metrics.
3. **Institutional knowledge.** The commit history of the prompt file documents the evolution of the product's behavioral specification—a record of what was tried, what failed, and what worked.
4. **Regression detection.** Automated evaluation runs on every prompt change detect regressions before they reach production.

A common anti-pattern is maintaining the system prompt in a configuration file, admin dashboard, or environment variable that is modified without version control.
This approach makes rollback difficult, eliminates review gates, and destroys the historical record of prompt evolution.
For any AI product where the system prompt materially affects behavior—which is to say, any AI product—the prompt should live in the codebase under version control.

### 2.1.5 Prompt length and the diminishing returns curve

System prompts consume tokens from the context window—tokens that could otherwise be used for retrieved content, tool-call history, or longer user queries.
There is therefore a tradeoff between prompt comprehensiveness (more instructions → better behavioral compliance) and context budget (more prompt tokens → fewer retrieval tokens).

Arxie's system prompt is approximately 290 tokens—modest by current standards.
This length was not reached by starting small and adding instructions; it was reached by starting large (approximately 800 tokens in early development) and pruning instructions that did not measurably improve evaluation metrics.

The pruning process revealed a diminishing returns curve:
- The first 100 tokens (role + core goals) accounted for approximately 60% of the behavioral improvement over a zero-instruction baseline.
- The next 100 tokens (tool-use policy) accounted for approximately 25% of additional improvement.
- The final 90 tokens (output constraints + edge cases) accounted for approximately 15% of additional improvement.

This distribution suggests that the most important prompt instructions are those that establish role, core behavioral goals, and tool-use policy.
Additional instructions for edge cases and formatting yield progressively smaller returns—and at some point, additional instructions can *degrade* performance by confusing the model or creating contradictory directives.

The practical implication for product managers: **measure the marginal impact of each prompt instruction.**
If an instruction does not measurably improve any product metric, it is consuming context budget without benefit and should be removed.
This empirical approach—adding, measuring, keeping or removing—is more reliable than intuition about what instructions "should" help.

## 2.2 Tool descriptions as UX

In an agent-based AI product, the model does not interact with tools directly—it interacts with *descriptions* of tools.
The tool description is the only information the model has about what a tool does, when to use it, and what to expect from it.
This makes tool description writing a user experience design problem, where the "user" is the model itself.

A tool description that seems clear to a human developer may be ambiguous, misleading, or incomplete from the model's perspective.
The consequences of poor tool descriptions are not cosmetic: they cause the agent to select the wrong tool, pass incorrect arguments, or skip tool use entirely—failures that cascade into incorrect outputs regardless of the model's underlying capability.

### 2.2.1 Tool selection as classification

When an agent receives a user query, it must decide which tool (if any) to invoke.
This decision is a classification problem: given the query and the available tool definitions, the model assigns the query to a tool (or to "no tool needed").

The features available for this classification are limited:
1. The tool's name
2. The tool's description
3. The tool's argument schema (parameter names, types, and descriptions)

Of these, the description carries the most information.
Tool names are typically short and may be ambiguous (`get_paper` vs. `get_paper_details`); argument schemas describe *how* to call a tool, not *when* to call it.
The description is where the model learns the tool's purpose, scope, and appropriate use cases.

This has a direct implication: **the quality of tool descriptions determines the accuracy of tool selection.**
A model that is highly capable of using tools correctly will nonetheless fail if it selects the wrong tool—and tool selection is governed almost entirely by descriptions. [Schick et al., 2023]

### 2.2.2 Three failure modes of tool descriptions

Tool descriptions fail in predictable ways.
Understanding these failure modes enables systematic diagnosis and correction.

**Under-specification.**
The description is too vague to discriminate between tools or between "use tool" and "don't use tool."

Consider a tool described as: *"Get information about a paper."*
This description could apply to retrieving metadata (title, authors, citation count), fetching the full text, finding papers that cite it, or summarizing its contents.
If the agent has multiple tools for these purposes, the vague description provides no basis for choosing among them.
The result is inconsistent tool selection: the same query may route to different tools on different runs, depending on which interpretation the model samples.

In Arxie, an early version of `get_paper_details` was described as *"Get details for a paper."*
This caused confusion with `read_paper_fulltext`, which also retrieves "details" in the colloquial sense.
The description was revised to specify the output type: *"Get detailed metadata for a specific paper by identifier... Returns JSON with normalized metadata and a formatted citation string."*
The explicit mention of "metadata" and "citation string" distinguishes it from full-text retrieval.

**Over-specification.**
The description is too narrow, causing the tool to be skipped when it should be used.

Consider: *"Use this tool ONLY when the user explicitly requests the PDF full text of a paper."*
This description will correctly trigger on "give me the full text of this paper" but will miss "what methods did they use?" or "how did they run the experiments?"—queries that require full-text access but do not mention PDFs.

Over-specification is often introduced as a "fix" for under-specification: the developer, observing that the tool is called too often, adds restrictive language.
This trades false positives for false negatives, which may be worse depending on the use case.
The correct fix is usually to clarify scope boundaries rather than to add restrictions.

**Scope overlap.**
Multiple tools have descriptions that could reasonably apply to the same query, and the model has no principled basis for choosing.

Arxie includes both `get_paper_details` and `get_paper` (an alias with identical functionality).
The descriptions are nearly identical:
- `get_paper_details`: *"Get detailed metadata for a specific paper by identifier..."*
- `get_paper`: *"Alias for get_paper_details. Get detailed metadata for a specific paper by identifier..."*

While the alias exists for developer convenience (some prompts may use "get paper" phrasing), the model sees two tools with overlapping descriptions.
This is harmless when both tools are functionally identical, but scope overlap between *different* tools causes unpredictable routing.

A subtler form of scope overlap occurs between tools that handle different cases of the same underlying intent.
In Arxie, `get_paper_full_text` returns plain text, while `read_paper_fulltext` returns structured sections (abstract, methods, results, discussion).
Both serve "get full text" intents, but for different downstream uses.
The descriptions must clarify when each is appropriate:
- `get_paper_full_text`: *"...extract its full text... Returns plain text."*
- `read_paper_fulltext`: *"Use this when the user asks for specific methodology, results, discussion details, or conclusions. Returns JSON with title, abstract, methods, results, discussion, and conclusion sections."*

The second description specifies the *trigger condition* (user asks for specific sections), not just the output format.

### 2.2.3 Encoding process instructions in descriptions

Section 2.1 established that the system prompt must encode the desired *process*, not just the desired output.
The same principle applies to tool descriptions: effective descriptions tell the model *when* to use the tool within the agent loop, not just what the tool does.

Arxie's `search_papers` tool is described as:
*"Search for relevant academic papers. **Use this first** to discover candidate sources. Returns JSON with a list of normalized paper metadata and citation strings."*

The phrase "Use this first" is not a capability description—it is a **process instruction**.
It tells the model that `search_papers` should be the initial action in most research queries, before `get_paper_details` or `read_paper_fulltext`.
This instruction directly supports the tool reliability constraint (≥0.90): without it, the model sometimes skips search entirely and generates responses from parametric knowledge.

Similarly, `read_paper_fulltext` includes:
*"Use this when the user asks for specific methodology, results, discussion details, or conclusions."*

This is a **trigger condition**: it specifies the user intents that should route to this tool.
The model learns not just what the tool does, but when to reach for it.

The pattern generalizes: for each tool, the description should answer three questions:
1. **What does it do?** (capability)
2. **When should I use it?** (trigger condition)
3. **Where does it fit in the workflow?** (process position)

A description that answers only the first question leaves the model to infer the second and third, which it will do inconsistently.

### 2.2.4 Tool descriptions and reliability

The reliability framework introduced by Rabanser et al. (2026) identifies *consistency* as a dimension distinct from accuracy: does the same input produce the same output across runs?
Tool descriptions directly affect consistency at the tool-selection layer.

Consider two semantically equivalent queries:
- "What methods did they use in this paper?"
- "How did they conduct the experiments?"

Both queries require `read_paper_fulltext` to answer properly.
If the tool description only mentions "methods," the first query may trigger the tool reliably while the second triggers it inconsistently—depending on whether the model interprets "conduct the experiments" as equivalent to "methods."

This is a form of **prompt robustness** (sensitivity to semantically equivalent rephrasings) applied to tool selection.
The description *"Use this when the user asks for specific methodology, results, discussion details, or conclusions"* attempts to cover multiple phrasings ("methodology," "results," "conclusions"), but cannot enumerate all possible rephrasings.

The practical mitigation is empirical testing: construct a set of paraphrased queries with the same expected tool call, measure the tool-call agreement rate, and revise descriptions to cover observed gaps.
This testing is analogous to prompt perturbation testing (Section 1.9), but applied to the tool-selection layer rather than the end-to-end output.

### 2.2.5 Token budget considerations

Tool definitions consume context window tokens.
Each tool contributes its name, description, and argument schema to every prompt, regardless of whether it is used.

For Arxie's seven tools, the token cost is approximately:

| Tool | Name | Description | Schema | Total |
|------|------|-------------|--------|-------|
| search_papers | 3 | 35 | 65 | ~103 |
| get_paper_details | 4 | 40 | 55 | ~99 |
| get_paper | 3 | 45 | 55 | ~103 |
| get_paper_full_text | 5 | 45 | 50 | ~100 |
| read_paper_fulltext | 4 | 55 | 50 | ~109 |
| get_paper_citations | 5 | 35 | 60 | ~100 |
| trace_influence | 4 | 45 | 85 | ~134 |
| **Total** | | | | **~748** |

These ~750 tokens are present in every agent invocation.
For a model with a 128K context window, this is negligible.
For a model with 8K context, it represents nearly 10% of available capacity—capacity that cannot be used for retrieved paper content or conversation history.

The tradeoff is between description richness (more tokens → better tool selection) and context availability (more tokens → less room for retrieval).
The diminishing returns principle from Section 2.1.5 applies: the first 20 tokens of a description (core capability) provide more marginal value than the next 20 tokens (edge case coverage).

A product manager should measure **tool-call precision per token**: if adding 15 tokens to a description improves tool-call precision from 0.88 to 0.94, that is 0.4 percentage points per token.
If adding another 15 tokens improves precision from 0.94 to 0.95, that is 0.07 percentage points per token—a 6× lower marginal value.
The second addition may not be worth the context budget.

### 2.2.6 Empirical testing of tool descriptions

Tool descriptions cannot be validated by inspection.
A description that seems clear to the developer may be ambiguous to the model; a description that seems overly verbose may be the minimum required for reliable routing.
The only reliable validation is empirical measurement.

**Constructing a tool-call evaluation set.**
The evaluation set consists of (query, expected tool calls) pairs:

```
Query: "Find papers about attention mechanisms in transformers"
Expected: [search_papers]

Query: "What methods did Vaswani et al. use in the original transformer paper?"
Expected: [search_papers, read_paper_fulltext]

Query: "Get me the citation info for arxiv 1706.03762"
Expected: [get_paper_details]

Query: "How has BERT influenced subsequent NLP research?"
Expected: [search_papers, get_paper_citations] or [trace_influence]
```

Note that some queries have multiple acceptable tool sequences; the evaluation should account for this.

**Measuring tool-call precision.**
Run the agent on each query with tool-call logging enabled.
Compute:
- **Tool-call precision**: (correct tool calls) / (total tool calls)
- **Tool-call recall**: (correct tool calls) / (expected tool calls)
- **Sequence accuracy**: (queries with correct tool sequence) / (total queries)

For Arxie, the tool-call evaluation set includes 40 queries across four categories: factual (single paper lookup), analytical (comparison or synthesis), exploratory (literature discovery), and deep (full-text required).
The current tool descriptions achieve 0.94 precision and 0.89 recall on this set.

**Iterative refinement.**
When tool-call failures are identified, the revision process is:
1. Examine the failed query and the incorrectly selected tool
2. Identify why the model made that selection (description ambiguity, scope overlap, missing trigger condition)
3. Revise the description to address the specific failure
4. Re-run the evaluation to confirm the fix and check for regressions

This is the same observe-diagnose-revise-evaluate loop used for system prompt refinement (Section 2.1.2), applied to tool descriptions.

### 2.2.7 Argument-level descriptions

Tool descriptions govern tool *selection*; argument descriptions govern argument *construction*.
The arguments the model passes to a tool are determined by the argument schema, including the descriptions of each field.

Compare two versions of a search query argument:

**Version A (minimal):**
```python
query: str = Field(..., description="Search query.")
```

**Version B (guided):**
```python
query: str = Field(
    ..., 
    description="Academic search terms. Include specific author names, "
                "paper titles, or technical concepts for better results. "
                "Example: 'attention mechanism transformer Vaswani'"
)
```

Version A provides no guidance on query construction.
The model will pass whatever phrasing seems reasonable, which may not align with how the underlying search API works (Semantic Scholar's relevance ranking, arXiv's query syntax).

Version B guides the model toward more effective queries: specific terms, author names, technical concepts.
This guidance propagates through the pipeline: better queries → better retrieval results → better final answers.

Argument descriptions are particularly important when:
1. The underlying API has non-obvious query syntax or ranking behavior
2. The argument affects downstream quality significantly (e.g., `limit` parameters that control how many results are fetched)
3. The argument has edge cases that the model might mishandle (e.g., identifier formats: DOI vs. arXiv ID vs. Semantic Scholar ID)

For Arxie's identifier arguments, the description explicitly lists accepted formats:
*"Paper identifier: Semantic Scholar paperId, DOI (optionally prefixed with DOI:), or arXiv id."*

This prevents failures where the model passes a paper title (not an identifier) or an incorrectly formatted DOI.

### 2.2.8 Summary

Tool descriptions are a critical UX surface in agent-based products.
They determine tool selection accuracy, which in turn determines whether the agent's capabilities can be reliably accessed.

The key principles:
1. **Descriptions are features for classification.** The model selects tools based on descriptions; poor descriptions cause misclassification.
2. **Three failure modes dominate.** Under-specification, over-specification, and scope overlap each require different remediation.
3. **Encode process, not just capability.** Descriptions should specify *when* to use the tool and *where* it fits in the workflow.
4. **Descriptions affect reliability.** Tool-call consistency is a measurable property that depends on description quality.
5. **Token budget constrains description length.** Measure marginal precision per token to optimize the tradeoff.
6. **Empirical testing is required.** Construct a tool-call evaluation set and measure precision/recall.
7. **Argument descriptions matter.** Field-level guidance affects argument construction and downstream quality.

The next section examines output formatting—the final transformation between agent behavior and user-facing response.

## 2.3 Output formatting as product design

Once an agent selects the right tools and retrieves relevant evidence, a second design problem emerges: how that evidence is packaged for consumption.
In conventional software products, formatting is often treated as a presentation-layer concern deferred to frontend development.
In AI products, this separation is weaker.
The model itself generates the primary output structure, and that structure directly affects trust, usability, and integration.

For this reason, output formatting should be treated as a product design problem, not a cosmetic post-processing step.
A product manager must specify output behavior with the same rigor used for capability requirements and model constraints.

### 2.3.1 Output format as interface contract

An AI product output is an interface between three parties:
1. The user (who reads and interprets the response)
2. The product logic (which may parse output for downstream actions)
3. External systems (APIs, exports, dashboards, workflow tools)

An output that is semantically correct but structurally inconsistent can fail all three interfaces.
A user may misinterpret an unsupported claim as evidenced;
a downstream parser may fail if section headers vary across runs;
an integration may break if key fields are omitted or renamed.

This implies that output format is part of the product contract.
In PRD terms, "the assistant answers correctly" is an incomplete requirement.
The complete requirement is: "the assistant answers correctly in a structure that supports verification and system interoperability."

Arxie illustrates this distinction.
Early outputs were free-form paragraphs with occasional inline citations.
Even when factual quality was acceptable, users reported low trust because they could not quickly map claims to evidence.
After introducing a structured answer format (claim blocks + inline citations + references section), perceived reliability increased despite marginal changes in underlying model accuracy.

The product implication is clear: **format quality can dominate perceived quality.**

### 2.3.2 The three-layer output model

A useful design abstraction is to decompose output into three layers:

**Semantic layer (what is being claimed).**
This layer contains the substantive content: conclusions, comparisons, caveats, and uncertainty.
Evaluation at this layer asks whether claims are true, relevant, and supported.

**Structural layer (how content is organized).**
This layer governs human readability: section order, bulleting, citation placement, reference grouping, and summary granularity.
Evaluation at this layer asks whether users can quickly locate key information and verify it.

**Operational layer (how output can be consumed by systems).**
This layer governs machine-readability: schema adherence, field completeness, stable key names, and error-state encoding.
Evaluation at this layer asks whether downstream systems can parse and process output reliably.

The layers are interdependent but not interchangeable.
A semantic improvement (better claim quality) does not automatically improve structural consistency or operational parseability.
Likewise, a perfectly valid JSON schema does not guarantee meaningful content.

For product design, this decomposition enables targeted interventions:
- Semantic failures → retrieval/prompt/model interventions
- Structural failures → format template and rendering interventions
- Operational failures → schema enforcement and post-processing interventions

### 2.3.3 Three enforcement mechanisms

Output formatting can be enforced through three mechanisms, each with different reliability/cost tradeoffs.

**Prompt-only enforcement.**
The prompt instructs the model to produce a specified structure (e.g., "Use headings: Summary, Evidence, References").

Advantages:
- Fast to implement
- No additional infrastructure
- Flexible to iterate

Limitations:
- No hard guarantees
- Sensitive to prompt drift and model updates
- High variance across runs for complex formats

Prompt-only enforcement is suitable for low-stakes formatting constraints (e.g., preferred tone, optional section order) but fragile for contractual outputs.

**Post-processing enforcement.**
The model output is transformed after generation: citations normalized, references deduplicated, missing sections inserted, ordering standardized.

Advantages:
- Deterministic normalization
- Can repair common formatting failures
- Reduces run-to-run variance

Limitations:
- Repair logic must be maintained
- Limited ability to recover missing semantic content
- Risk of over-correction if parser assumptions are brittle

Arxie uses post-processing for citation normalization and references rendering.
This reduced formatting variance materially without changing model weights.

**Schema-constrained generation.**
The model is required to emit a structured object (e.g., JSON schema with required fields).

Advantages:
- Strongest structural guarantees
- Directly compatible with downstream systems
- Easier automated testing

Limitations:
- Increased prompt/tooling complexity
- Potential reduction in expressive flexibility
- Model/provider support varies

For high-stakes product paths (programmatic consumption, compliance workflows), schema-constrained generation is usually the only defensible option.

A practical PM rule:
- If format errors are recoverable and low-cost → prompt-only or lightweight post-processing
- If format errors break workflows or trust → schema constraints + deterministic post-processing

### 2.3.4 Citation formatting as trust infrastructure

In research products, citations are not ornamental.
They are the primary mechanism through which users audit claims.
Formatting decisions therefore affect epistemic trust directly.

A useful citation architecture contains three elements:

1. **Local evidence links (inline citations).**
Each non-trivial claim should be locally anchored with citation tokens (e.g., Author et al., Year).
Without local anchors, users must infer which evidence supports which claim.

2. **Global source index (references section).**
All cited works must be listed in a stable format with sufficient identifiers (DOI, arXiv ID, paperId) for retrieval.
Without global indexing, inline citations become unverifiable labels.

3. **Provenance metadata (operational layer).**
For system-level auditing, each citation should map to source metadata and retrieval events (where available).
This supports debugging when citations appear inconsistent or stale.

Arxie's output design evolved along this path:
- Phase 1: free-text with occasional citations
- Phase 2: mandatory inline citations + references section
- Phase 3 (in progress): structured citation objects in machine-readable output

This progression reflects a general maturity pattern:
as products move from interactive exploration toward professional workflows, citation format must shift from human-readable convention to machine-auditable contract.

### 2.3.5 Failure modes and product consequences

Output formatting failures should be classified by business impact, not only technical severity.

**Citation drift.**
A citation appears near a claim but actually supports a different statement.
Consequence: users over-trust unsupported claims.

**Orphan claims.**
Substantive claims appear without evidence anchors.
Consequence: verification cost increases; trust decreases.

**Reference hallucination.**
References list contains non-existent or mismatched sources.
Consequence: catastrophic trust failure in academic contexts.

**Schema breakage.**
Required fields omitted or renamed in machine-readable output.
Consequence: integration failures, downstream pipeline crashes.

**Intra-run inconsistency.**
The same entity is formatted differently within one response (e.g., two citation styles).
Consequence: perceived low quality and ambiguity.

**Inter-run inconsistency.**
Equivalent queries produce structurally different outputs across runs.
Consequence: difficult automation and regression detection.

PM response should map failure mode to action:
- Trust-critical failures (reference hallucination, orphan claims in high-stakes mode) → hard release blockers
- Workflow-critical failures (schema breakage) → integration blockers
- Cosmetic inconsistencies → backlog unless they materially affect user behavior

### 2.3.6 PRD-level output contracts

PRDs for AI products should define output contracts explicitly.
A useful contract specification includes:

**Required fields (operational):**
- `answer`
- `references[]`
- `status`
- `errors[]` (if applicable)

**Recommended fields (trust + observability):**
- `claims[]`
- `citations[]`
- `confidence` (if calibrated and validated)
- `provenance` (source identifiers, retrieval metadata)

**Behavioral constraints:**
- Every non-trivial claim must have at least one citation
- References must be deduplicated and resolvable
- If evidence is insufficient, response must explicitly state limitations

**Fallback behaviors:**
- No-results state
- Partial-results state
- Retrieval-failure state

**Versioning policy:**
- Output schema versions must be backward-compatible for one deprecation window
- Breaking changes require migration notes and parser updates

This shifts output design from informal prompt wording to explicit product contract management.

### 2.3.7 Evaluation strategy for formatting quality

Formatting quality requires dedicated evaluation metrics separate from semantic correctness.
A minimal evaluation suite includes:

**Schema adherence rate.**
Fraction of responses that satisfy required schema fields/types.

**Citation linkage precision.**
Fraction of cited claims where citations correctly support the associated claim.

**Orphan-claim rate.**
Fraction of non-trivial claims without citations.

**Reference resolvability rate.**
Fraction of references that resolve to valid identifiers/sources.

**Run-to-run format consistency.**
Structural similarity across repeated runs on identical inputs.

**Fallback correctness.**
Fraction of failure scenarios where output uses the correct failure schema/state.

These metrics should run in CI alongside semantic evaluation.
A common anti-pattern is to evaluate format manually during development and only automate semantic tests.
This creates regression risk: minor prompt changes can silently degrade structural reliability while semantic scores remain stable.

For Arxie, formatting regression tests should be tied to both prompt and tool-description changes, since both can alter output structure indirectly.

### 2.3.8 Summary

Output formatting is where model capability becomes product value.
An AI system that is semantically strong but structurally inconsistent is difficult to trust and difficult to integrate.

The core design principles are:
1. Treat output format as an interface contract
2. Separate semantic, structural, and operational concerns
3. Choose enforcement mechanism by failure cost
4. Design citations as trust infrastructure, not presentation detail
5. Classify format failures by product impact
6. Encode output contracts explicitly in the PRD
7. Evaluate formatting quality with dedicated, automated metrics

The next section addresses context window budget management, where formatting decisions intersect directly with token allocation and latency/cost constraints.

## 2.4 Context window budget management

Context engineering is constrained by a finite resource: tokens.
Every instruction, tool definition, retrieved passage, intermediate trace, and generated response competes for space in the model context window and for inference budget.
For agentic systems, this is not a secondary optimization issue.
It is a core product design constraint that determines quality, latency, and cost simultaneously.

In conventional software systems, memory management is largely invisible to product management.
In LLM systems, context allocation is directly tied to user-facing outcomes.
If too few tokens are allocated to retrieved evidence, answer quality degrades.
If too many tokens are allocated to retrieval traces and tool logs, latency and cost increase non-linearly.
If system instructions are too long, critical evidence is pushed out of context.

For this reason, context budgeting should be treated as a first-class product mechanism with explicit design rules, operational limits, and measurement.

### 2.4.1 The context budget as an optimization problem

For each query, an agent must allocate a fixed token budget across competing components:

\[
B = B_{sys} + B_{tools} + B_{user} + B_{history} + B_{retrieval} + B_{scratch} + B_{output}
\]

Where:
- \(B_{sys}\): system prompt tokens
- \(B_{tools}\): tool schema/description tokens
- \(B_{user}\): user query tokens
- \(B_{history}\): conversation history tokens
- \(B_{retrieval}\): retrieved evidence tokens
- \(B_{scratch}\): intermediate reasoning / tool traces
- \(B_{output}\): generated response tokens

Even with large-context models, this allocation remains binding because cost and latency scale with total processed tokens.
In multi-turn agent loops, total processed tokens are often much larger than the maximum instantaneous window due to repeated re-ingestion of prior context.

A practical optimization objective is:

\[
\max \text{AnswerUtility}(B_{retrieval}, B_{output}) \quad \text{s.t.} \quad
\text{Latency}_{p95} \leq L_{max}, \ \text{Cost/query} \leq C_{max}
\]

This explicitly frames context allocation as a constrained product optimization problem, not an engineering afterthought.

### 2.4.2 Why token costs compound in agent loops

A common early-stage mistake is estimating token cost from the final prompt only.
This underestimates cost and latency for ReAct-style agents.

In a single-pass chatbot, one request and one response dominate cost.
In an agent loop, each iteration includes:
1. Previous messages
2. Tool invocation instructions
3. Tool output observations
4. New model reasoning and next action

Thus, iteration \(t\) reprocesses most of \(t-1\) plus new content.
For a query requiring five tool calls, total processed input tokens may be 3–10× the apparent "final context size."

In Arxie, deep literature queries often follow this pattern:
- Turn 1: search papers
- Turn 2: inspect paper details
- Turn 3: fetch citations / full text sections
- Turn 4: compare evidence
- Turn 5: synthesize final answer with references

If each turn adds 1,000–2,000 tokens of observations and prior turns are retained verbatim, cumulative input tokens increase super-linearly.
This is the primary driver of unexpected cost spikes in agentic products.

Product implication: **query complexity tiers must be part of the PRD and pricing model.**
A flat cost assumption per query is structurally wrong for agent workflows.

### 2.4.3 Arxie budget decomposition

Arxie currently operates with the following approximate token composition per standard query:

| Component | Typical Tokens |
|-----------|----------------|
| System prompt | 250–350 |
| Tool definitions | 700–900 |
| User query | 30–150 |
| Retrieval snippets (2–5 papers) | 800–2,500 |
| Intermediate traces | 300–1,500 |
| Output | 250–800 |
| **Total processed (single pass equivalent)** | **2,300–6,200** |

For deep multi-hop queries, total processed tokens across turns can exceed 15,000–40,000, depending on tool-call depth and full-text usage.

Two observations follow:
1. Tool definitions are a fixed tax per call. This reinforces Section 2.2: verbose tool schemas consume persistent budget and should be justified by measurable routing gains.
2. Retrieval snippets dominate variable cost. Retrieval quality and compression strategy have higher leverage on cost/latency than small prompt edits.

### 2.4.4 Budget policies: fixed, adaptive, and tiered

Three policy families are commonly used.

**Fixed budget policy.**
A single cap is applied to all queries (e.g., max 4,000 input tokens, max 600 output tokens).

Advantages:
- Simple to reason about
- Predictable cost envelope

Limitations:
- Under-allocates complex queries
- Over-allocates simple queries
- Degrades either quality or efficiency depending on chosen cap

Suitable for MVPs and low-variance workloads.

**Adaptive budget policy.**
Budget is adjusted dynamically from query features (query length, ambiguity, detected task type, retrieval confidence).

Advantages:
- Better cost-quality tradeoff
- Higher efficiency across heterogeneous workloads

Limitations:
- Requires complexity estimator
- Harder to debug and explain

Suitable once telemetry is available.

**Tiered policy.**
Discrete modes with explicit product semantics (e.g., "Quick answer" vs. "Deep research").

Advantages:
- User-visible control over quality/cost/latency
- Simpler than full adaptivity
- Enables clear SLAs per mode

Limitations:
- Requires good mode defaults
- Risk of user confusion if differences are unclear

Arxie should use tiered policy as default:
- **Standard mode**: fast synthesis from abstracts/metadata, limited tool depth
- **Deep mode**: full-text section retrieval, broader citation chasing, larger output allowance

This aligns model behavior with user expectations and with business cost controls.

### 2.4.5 Retrieval compression and evidence selection

Increasing retrieval tokens improves factual grounding only up to a point.
Beyond that point, marginal evidence quality declines while token cost and distraction increase.

Effective context budgeting therefore requires evidence compression policies:

1. **Relevance-first truncation.** Rank passages by semantic relevance and keep top-k under token cap.
2. **Field-aware extraction.** Include abstract/methods/results snippets rather than full paper text by default.
3. **Redundancy suppression.** Remove near-duplicate evidence across papers.
4. **Citation-prioritized retention.** Retain passages with high citation utility (clear claims, methods, quantitative results).

Arxie's `read_paper_fulltext` already extracts structured sections (abstract, methods, results, discussion, conclusion).
This is a strong compression primitive: section extraction reduces raw full-text load while preserving semantically critical evidence.

A practical PM heuristic:
- If task requires broad survey → prioritize abstract + conclusion snippets across many papers
- If task requires method critique → prioritize methods + results from fewer papers

The same token budget can support different evidence strategies depending on user intent.

### 2.4.6 Conversation history management

Multi-turn memory competes directly with evidence budget.
In long sessions, retaining full history can displace retrieval context and degrade answer grounding.

History policies:

**Full retention.** Keep all prior turns.
- High coherence
- Rapid token growth
- Poor scalability

**Windowed retention.** Keep last N turns.
- Predictable token usage
- Risk of losing long-range constraints

**Summarized retention.** Periodically summarize prior turns into compact state.
- Strong token control
- Summary quality risk (loss or distortion)

For research assistants, summarized retention is typically superior when paired with explicit state fields:
- Active question
- Included/accepted sources
- Excluded sources
- Outstanding uncertainties
- Formatting constraints

This preserves decision-critical context while controlling growth.

### 2.4.7 Output budget and structural tradeoffs

Output tokens are often treated as a residual budget.
This is a mistake in products where output structure is part of the trust mechanism.

In Arxie, references, caveats, and evidence-linked claims are not optional verbosity.
If output caps are too low, these sections are truncated first, causing trust regressions even when core answer text is intact.

Therefore output budgeting should reserve structural minima:
- Minimum citation slots
- Minimum references section capacity
- Minimum uncertainty/caveat space for low-evidence cases

A practical output policy:
1. Reserve fixed tokens for structure (references + caveat fields)
2. Allocate remainder to narrative synthesis
3. If remaining budget is insufficient, reduce narrative length before dropping evidence structures

This enforces product priorities under tight budgets.

### 2.4.8 Latency and concurrency implications

Context budgeting affects not only per-query latency but system throughput.
Higher token loads increase model inference time and can reduce effective concurrency under provider rate limits.

Given arrival rate \(\lambda\) and mean service time \(W\), concurrency load follows Little's Law (\(L = \lambda W\)). [Jain, 1991]
If aggressive context budgets increase \(W\), the same traffic requires higher concurrency, increasing queueing delays and tail latency.

Thus token budget decisions should be validated against p95 latency targets under projected traffic.
This links context engineering directly to production readiness constraints (Chapter 1.9).

For Arxie, deep mode should be treated as a bounded-capacity path:
- lower allowed concurrency
- explicit user feedback ("deep analysis may take longer")
- optional asynchronous delivery for very large evidence sets

### 2.4.9 Budget observability and control loops

Context budgeting must be instrumented.
Without telemetry, budget policy becomes guesswork.

Minimum metrics:
- Input tokens by component (system/tools/history/retrieval)
- Output tokens
- Tool-call depth
- Cost/query by mode
- Latency by mode and complexity tier
- Truncation events (which component was truncated)

Control loop:
1. Observe budget metrics and failure patterns
2. Identify bottleneck (e.g., retrieval overrun, history bloat, output truncation)
3. Adjust policy (caps, compression, mode defaults)
4. Re-evaluate quality + latency + cost

This loop should run continuously in production.
Static budgets tuned on offline datasets drift as user behavior changes.

### 2.4.10 PRD requirements for context budgeting

A robust PRD should include explicit context budget requirements:

- **Per-mode token caps** (input/output)
- **Maximum tool depth** by mode
- **History retention policy** (windowed/summarized)
- **Evidence selection policy** (section-level, top-k, dedup)
- **Structural output minima** (references/citations/caveats)
- **Latency and cost targets per mode**
- **Fallback behavior when caps are hit** (summarize, ask follow-up, switch to async)

This prevents implicit budget assumptions from leaking into production behavior.

### 2.4.11 Summary

Context window management is the mechanism through which AI products trade off quality, speed, and cost.
For agentic systems, token usage compounds across tool loops, making budget policy a central product decision.

The core principles:
1. Treat context allocation as constrained optimization, not prompt tuning
2. Model cumulative token costs across agent turns
3. Use tiered or adaptive budgets for heterogeneous query complexity
4. Prioritize evidence compression over indiscriminate truncation
5. Reserve output budget for trust-critical structure
6. Instrument token flows and run a continuous policy control loop
7. Encode budgeting rules explicitly in the PRD

The next section examines the prompt engineering ceiling: where context and instruction design stop delivering returns and model customization becomes necessary.

## 2.5 The prompt engineering ceiling

Prompt engineering is the highest-leverage early intervention in an AI product because it converts an under-specified system into a specified one. In Arxie, the system prompt in `src/ra/agents/research_agent.py` encodes citation rules, retrieval requirements, and full-text triggers; the tool schemas in `src/ra/tools/retrieval_tools.py` provide the routing hints that make those instructions operational. This is why early prompt work often produces the fastest visible gains: it does not create new capability, but it reduces the amount of behavior left to implicit model interpretation. [Ouyang et al., 2022; Reynolds & McDonell, 2021]

### 2.5.1 Problems prompts fix well

Prompts are well suited to three classes of problems: missing format constraints, weak workflow cues, and inconsistent response structure. Arxie's evolution from loose "cite sources" behavior to mandatory inline `(Author et al., Year)` citations, explicit `References` sections, and the instruction `You MUST call search_papers at least once before finalizing any answer` belongs to this category. These interventions constrain how existing capability is deployed; they do not alter the model weights or expand the system's access to information. [Ouyang et al., 2022; Zamfirescu-Pereira et al., 2023]

The practical test is metric movement after the prompt change. If citation density increases, tool-call skip rate declines, or output structure becomes more consistent, the prompt change is doing real product work rather than cosmetic cleanup. Arxie's baseline report is consistent with this pattern: citation precision passes its target at `0.8600`, and tool success reaches `0.9982`, indicating that prompt and tool-description work can materially improve grounded behavior when the surrounding system is already capable of retrieval and citation formatting.

### 2.5.2 Problems prompts influence only probabilistically

The second category contains behaviors that prompts can bias but not guarantee. Tool selection, abstention, ambiguity handling, and response length all fall into this group. Arxie's instructions about when to call `read_paper_fulltext`, when to avoid answering from prior knowledge, and how to structure output increase the probability of grounded responses, but they remain probabilistic controls. Long contexts, ambiguous queries, or noisy retrieval results can still produce violations. [Rabanser et al., 2026]

This is the operational meaning of a prompt ceiling: prompt quality changes the distribution of behaviors, not the existence of hard guarantees. A team that treats prompt edits as deterministic fixes will systematically overestimate reliability. The correct workflow is therefore experimental. Prompt changes should be proposed as hypotheses, measured against a fixed evaluation set, and retained only if they improve product metrics under repeatable conditions. [Jain, 1991; Rabanser et al., 2026]

### 2.5.3 Problems prompts cannot solve

The hard ceiling appears when the failure requires external truth, deterministic enforcement, or new system capability. A prompt cannot tell the model whether a citation exists in Semantic Scholar; it can only instruct the model to behave as if existence matters. Arxie's product differentiation in the PRD—live retrieval, PDF parsing, citation verification, confidence scoring, and recent-paper coverage—depends on tools and post-processing rather than wording alone. The same principle applies to citations: `src/ra/citation/formatter.py` can normalize how citations are rendered, but prompt instructions alone cannot verify whether a cited source actually supports the claim. [Sculley et al., 2015]

Operational constraints sit beyond the prompt layer as well. A prompt cannot reduce p95 latency, absorb provider rate limits, or create access to recent papers without a retrieval pipeline. Arxie's baseline evaluation makes this distinction concrete: citation precision and tool success pass target, but p95 latency remains `55.3008s`, far above the intended interactive threshold. That shortfall belongs to system design, retry policy, context budgeting, and infrastructure decisions rather than to missing prompt wording. [Jain, 1991]

### 2.5.4 A decision rule for escalation

Once prompt iteration stops moving the relevant metric, the next question should not be "what wording have we not tried?" but "which layer owns the failure?" A practical decision rule is:

| Failure class | Arxie example | Prompt leverage | Next layer |
|---|---|---|---|
| Format compliance | Inline citation wording and answer structure | High | Prompt and tool-description revision |
| Process consistency | Correct tool order and full-text triggers | Medium | Prompt revision plus evaluation |
| Verification | Citation existence and claim support | Low | Post-processing, validators, release gates |
| Capability and operations | Full-text access, latency, rate limits, recent-paper coverage | Minimal or none | Retrieval architecture, caching, routing, operating policy |

This rule has two implications for the rest of the book. First, prompt work without measurement is not disciplined engineering, because the ceiling can only be located by repeated evaluation. That is the subject of Chapter 3. Second, many apparent prompt failures are misclassified architecture or operations problems, which is why the Arxie example turns next to evaluation, then to agent design and shipping, rather than immediately to fine-tuning. Model customization remains relevant, but it is the last escalation step after the prompt, system, and architecture layers have been stabilized. [Rabanser et al., 2026; Sculley et al., 2015]

## 2.6 Feature scoping at the prompt layer

The next section of this chapter will formalize a scoping rule that should be applied before implementation work begins: every requested feature should be classified as a prompt-layer change, a system-layer change, or an architecture-layer change. The purpose is not taxonomy for its own sake; it is to prevent wasted iteration on the wrong layer.

### 2.6.1 Prompt-layer changes

This section will cover feature requests that mainly change wording, ordering, or behavioral emphasis:

- answer structure and tone
- citation wording and density
- tool-use reminders and trigger conditions
- lightweight refusal and uncertainty phrasing

### 2.6.2 System-layer changes

This section will cover requests that need deterministic enforcement or validation:

- citation verification
- fallback states and no-results handling
- output normalization and schema enforcement
- truncation control and failure-state formatting

### 2.6.3 Architecture-layer changes

This section will cover requests that require new capabilities or new data flow:

- full-text parsing
- multi-hop retrieval
- memory and session state
- proposal workflows and evidence reuse

### 2.6.4 Arxie mapping

The Arxie case study will use the following mapping:

- citation formatting: prompt and formatter work
- hallucination prevention: retrieval, verification, and evaluation work
- deep research behavior: retrieval and tool architecture work
- fine-tuning decisions: only after the earlier layers stabilize and metrics plateau

## References

- Jain, R. (1991). *The Art of Computer Systems Performance Analysis: Techniques for Experimental Design, Measurement, Simulation, and Modeling*. Wiley.
- Ouyang, L., Wu, J., Jiang, X., et al. (2022). Training Language Models to Follow Instructions with Human Feedback. *NeurIPS*.
- Rabanser, S., Kapoor, S., Kirgis, P., Liu, K., Utpala, S., & Narayanan, A. (2026). Towards a Science of AI Agent Reliability. *arXiv:2602.16666*.
- Reynolds, L., & McDonell, K. (2021). Prompt Programming for Large Language Models: Beyond the Few-Shot Paradigm. *CHI Extended Abstracts*.
- Schick, T., Dwivedi-Yu, J., Dessì, R., et al. (2023). Toolformer: Language Models Can Teach Themselves to Use Tools. *NeurIPS*.
- Sculley, D., Holt, G., Golovin, D., et al. (2015). Hidden Technical Debt in Machine Learning Systems. *NeurIPS*.
- Zamfirescu-Pereira, J. D., Wong, R. Y., Hartmann, B., & Yang, Q. (2023). Why Johnny Can't Prompt: How Non-AI Experts Try (and Fail) to Design LLM Prompts. *ACM CHI*.
- Zheng, L., Chiang, W.-L., Sheng, Y., et al. (2024). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. *NeurIPS*.
