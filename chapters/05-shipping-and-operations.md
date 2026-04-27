# Chapter 5: Shipping and Operating AI Products

> Status: outline anchored to Arxie's PRD, deployment posture, wrappers, and release checks.

This chapter addresses the gap between a system that works in evaluation and a system that can be operated repeatedly under real constraints. Arxie is useful here because its repo already exposes cost limits, network constraints, deployment wrappers, release checks, and baseline operational risks.

**Primary Arxie artifacts**

- `docs/PRD.md`
- `Dockerfile`
- `docker-compose.yml`
- `Makefile`
- `scripts/with-env.sh`
- `tools/paperclip-quality-preflight.sh`
- `src/ra/utils/config.py`

## 5.1 The demo-to-production gap

This section will explain why AI products often demo well and fail in regular use, and why evaluation alone is necessary but still insufficient.

## 5.2 Runtime constraints and infrastructure posture

This section will cover API dependencies, network constraints, environment setup, retries, and deployment surfaces.

## 5.3 Monitoring AI behavior

This section will examine which signals matter in production: tool-call patterns, output structure, latency, cost, and quality drift.

## 5.4 Cost and latency control

This section will connect prompt size, retrieval depth, retries, and model routing to the operating margin of the product.

## 5.5 Release discipline

This section will show how checklists, preflight scripts, and release gates prevent false confidence from demos and isolated local runs.

## 5.6 Iteration velocity

This section will cover how a team keeps improving the system without breaking it: short eval loops, small deltas, and artifact-backed debugging.

## 5.7 Provider drift and rollback

This section will address silent provider changes, fallback models, rollback policy, and why the evaluation stack doubles as an operational contract.
