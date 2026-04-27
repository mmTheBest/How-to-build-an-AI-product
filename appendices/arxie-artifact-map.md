# Appendix: Arxie Artifact Map

This appendix maps the book's major arguments to concrete artifacts in the Arxie repository so readers can connect abstract product reasoning to implemented systems.

| Book area | Primary Arxie artifacts | Why they matter |
|---|---|---|
| Model selection | `docs/PRD.md`, `docs/eval-baseline.md` | Defines product constraints, targets, and observed baseline outcomes |
| Context engineering | `src/ra/agents/research_agent.py`, `src/ra/tools/retrieval_tools.py` | Shows the system prompt, tool-use rules, and tool descriptions that shape behavior |
| Output and citations | `src/ra/citation/formatter.py` | Demonstrates that trust-critical formatting requires deterministic support, not only prompt wording |
| Evaluation | `tests/eval/harness.py`, `src/ra/eval/release_gate.py` | Converts product requirements into repeatable tests and release decisions |
| Architecture | `src/ra/retrieval/`, `src/ra/parsing/pdf_parser.py`, `src/ra/proposal/` | Shows how retrieval, parsing, and workflow state expand capability beyond prompting |
| Operations | `Dockerfile`, `scripts/with-env.sh`, `Makefile`, `tools/paperclip-quality-preflight.sh` | Exposes deployment posture, environment constraints, and release discipline |

## Reading Note

The manuscript uses Arxie as a stable example of an academic research assistant. Forward-looking platform-expansion material from the example repo should be treated as roadmap context rather than as the mainline example for Chapters 1-5.
