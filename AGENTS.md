# AGENTS.md

This repository is used with Codex and OpenClaw in a codex-only setup.

## Context Order

1. Read `README.md` first if it exists.
2. Read task-relevant docs under `docs/`, `paper/`, or similar project folders.
3. If `CLAUDE.md` exists, treat it as supplemental legacy context rather than the primary instruction file.

## Working Rules

- Keep changes minimal and scoped to the requested task.
- Match the existing code style, structure, and tooling already used in the repo.
- Prefer project-native scripts and package managers over inventing new workflows.
- Do not edit unrelated files or introduce broad refactors without explicit need.
- Surface missing credentials, services, or environment dependencies instead of faking success.

## Verification

- Run the smallest relevant verification step for the files you changed.
- Prefer repo-native commands such as test, lint, typecheck, build, or targeted scripts.
- If no automated verification exists, say that clearly and provide a concrete manual check.

## Codex/OpenClaw Notes

- `AGENTS.md` is the primary cross-tool context file for this repo.
- Avoid Claude-specific wrapper paths or assumptions in this repository unless they are explicitly still maintained.
- Keep any legacy `CLAUDE.md` content aligned with this file if both are used.
