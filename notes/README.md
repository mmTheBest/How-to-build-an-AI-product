# Notes (Working Material)

This directory contains working notes for *How to Build an AI Product From Scratch*.

- Content in `notes/` is **not** considered publishable.
- The publishable manuscript lives in `book/src/`.
- Notes may arrive out of order relative to the book structure.

## Daily workflow

1. Capture new input as-is in `notes/inbox/YYYY-MM-DD.md`.
2. Optionally extract reusable fragments into `notes/snippets/`.
3. Integrate relevant material into `book/src/...` (edited, consistent terminology, impersonal academic tone).
4. Record non-trivial choices in `notes/decisions.md`.

## Tagging conventions

Use the following lightweight metadata on each note item:

- `tags`: comma-separated keywords (e.g., `rag`, `eval`, `kg`, `latency`, `cost`, `tooling`, `failure-modes`, `compliance`).
- `maps_to`: tentative target in the book (e.g., `Part 1 / Ch 01 / Metrics`).
- `status`: `raw` | `edited` | `integrated`.

## Inbox template

```md
# YYYY-MM-DD — Inbox

## Item 1 — Title (optional)
(text)

- tags: ...
- maps_to: ...
- status: raw

## Open questions / TODO
- ...
```
