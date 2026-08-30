# GapFinder

GapFinder is a stateful literature agent for one narrow field: animal models of heart
failure with preserved ejection fraction (HFpEF). It maintains a coverage map of
species × disease model × outcome domain, rebuilt from every paper it has read, and keeps
that map between runs. It alerts only when the map's state changes — a verified gap
receives its first paper, two papers begin disagreeing about the same measure, or a cell
that held only characterisation studies gains its first interventional study.

## What makes it different

Existing tools infer gaps from text similarity at the moment you prompt them. GapFinder
computes coverage explicitly and reports it with a denominator: not "under-studied" but
**"0 of 42 screened"** — 42 papers sat at that species × model pair and none of them
reported that outcome domain. The unit of alerting is a knowledge-state change, not a
keyword match. The agent runs unattended on a schedule and holds the previous cycle's map
in object storage, so "what changed" is a diff between two computed maps rather than a
fresh query against a search index.

## Live demo

https://gapfinder-dashboard-588003694253.us-east1.run.app

## Stack

- **Gemini 3.5 Flash-Lite** via the **Google GenAI SDK** — per-abstract structured
  extraction with a Pydantic response schema and a verbatim-or-null rule
- **Cloud Run Job** — the daily agent cycle
- **Cloud Run Service** — the read-only dashboard
- **Cloud Scheduler** — 07:00 America/New_York, no human in the loop
- **Cloud Storage** — all cycle state and timestamped history
- **Secret Manager** — both API keys, injected at runtime, never in the image
- **Cloud Build** and **Artifact Registry** — both images

## Validation

The demo replays genuinely held-out papers: the corpus is frozen at 2025-08-31 and the
replay set covers 2025-09 → 2026-08, never seen when the grid was built, with zero
overlap between the two sets. The grid separates four cell states — evidence,
no_evidence_found, unscreened, and not_applicable — so silence is only reported as a gap
when enough papers were screened for that silence to mean something; the fourth state
caught a real species-assignment bug on its first run.

Built solo by a cardiovascular postdoc on his own literature.
