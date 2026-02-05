# Standard workflow (writing-first playbook)

This document defines the repeatable process used to produce reviewable manuscript updates.

## Inputs
- Target chapter/section.
- Daily quota: default deliverable is **~3,000 words/day** unless Mya approves a smaller scope.
- Constraints: impersonal academic English; sentence-level citations; papers/books only unless explicitly allowed.

## Backend subtasks (must be run each time)

The writing procedure is a sequence of major steps; each step includes explicit verification/correction rules. “Correction” is not a separate phase; it is embedded as required checks within each step.

### 1) Micro-outline (10–20 min)
**Goal:** refine scope below chapter level (subsections) and define the teaching intent and internal logic before drafting prose.

**Write:**
- Thesis (1–2 sentences) stated in terms of the *general playbook objective* (i.e., what decision/procedure the reader learns), not the running example.
- Bullet outline including:
  - definitions
  - procedure/algorithm
  - failure modes
  - reader artifacts (what the reader can produce)
  - transitions to adjacent sections/chapters

**Verification rules (must pass before Step 2):**
- **Objective-first structure:** headings and thesis statements must serve the playbook objective (teach a transferable decision/procedure). Examples are included only to illustrate the objective, and should not become the main subject.
- **Reader-expectation format:** headings should follow conventional technical expository structure (problem → method/procedure → example → limitations), avoiding unexpected rhetorical section labels.
- **Scope check:** the outline can be completed within the planned timebox.
- **Blueprint consistency:** the outline does not introduce concepts that belong to later chapters without an explicit forward reference.
- **Non-vagueness check:** every bullet is phrased as a claim, definition, step, or failure mode (no placeholders like “talk about X”).

### 2) Reference pack (30–90 min)
**Goal:** ground the draft in reliable sources and prevent unsupported claims.

**Write:**
- Gather 6–15 references (papers/books unless explicitly allowed otherwise).
- For each reference:
  - extract (paraphrase) the specific claim(s) it supports
  - record the limitation/boundary conditions relevant to the manuscript
- Maintain a mapping: planned paragraph/sentence → citation(s).

**Verification rules (must pass before Step 3):**
- **No silent skipping rule:** if reference access is blocked (network, paywall, tooling) or unclear, stop and ask Mya immediately; do not proceed with under-sourced writing.
- **Reference relevance:** each reference is used for a specific claim (no decorative citations).
- **Reference correctness:** claim attribution matches the reference’s stated result/argument.
- **Coverage check:** every non-trivial planned claim has at least one supporting reference or is explicitly labeled as heuristic and deferred.
- **Plagiarism safety:** no copied phrasing is stored in the notes; only paraphrased claims + bibliographic info.

### 3) Deep reasoning synthesis (30–60 min)
**Goal:** connect references into a coherent argument rather than a citation list.

**Write:**
- Connect references into an integrated line of reasoning:
  - identify where sources agree/conflict
  - state assumptions and boundary conditions
  - convert informal advice into operational definitions
  - enumerate trade-offs and implications

**Verification rules (must pass before Step 4):**
- **Logic-flow check:** the argument can be read as a sequence of “because … therefore …” statements.
- **No exaggeration:** conclusions are bounded to what the references justify.
- **Inconsistency scan:** explicitly resolve or flag disagreements across sources.

### 4) Draft in portions (60–180 min)
**Goal:** produce prose in stable blocks with local completeness.

**Write:**
- Draft the section in 2–4 blocks; each block must include:
  - explicit definitions
  - procedures expressed as steps or pseudo-algorithms
  - measurable targets and measurement protocols
  - RA illustration only where it clarifies the concept

**Verification rules (must pass for each block before continuing):**
- **Depth/coverage gate:** each block must add substantive explanation (definitions + rationale + implications); if the block is only a short checklist without explanation, expand or split the task into smaller subtasks and gather more sources before proceeding.
- **Sentence-level citation check:** every non-trivial sentence has citation(s).
- **Vague-claim check:** replace adjectives (e.g., “robust”, “fast”) with measurable definitions.
- **Reference-to-sentence audit:** citations attached are appropriate for the exact claim.
- **Continuity check:** transitions connect the block to the previous block without jumps.

### 5) Integrated revision pass (20–60 min)
**Goal:** produce a reviewable manuscript-quality draft.

**Checks and corrections (all mandatory):**
1. **Logic/consistency:** no contradictions with existing blueprint or earlier chapters.
2. **No exaggeration:** qualify claims; state limitations.
3. **Plagiarism safety:** ensure no sentence closely mirrors a single source; paraphrase and attribute.
4. **Reference correctness:** sentence-level citations are correct and sufficient.
5. **Style pass:** impersonal academic English, consistent terminology, smooth transitions.

### 6) Review packet (what is sent to Mya)
- Scope: what file/section changed.
- Summary: what was added/changed (content-level only).
- Open questions (if any).
- The draft text (or a diff).
- References list.
- Ask whether to proceed.

## Outputs
- Manuscript update in `book/src/...`.
- Reference pack note in `notes/sources/...` (internal).
