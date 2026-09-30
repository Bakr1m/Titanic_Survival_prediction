# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [1.0.0] - 2026-09-30

### Added
- Random Forest survival classifier (test ROC-AUC 0.8435, F1 0.7383) with
  sklearn preprocessing pipeline, saved as `models/titanic_pipeline.joblib`
  and published as release asset `v1.0.0`.
- FastAPI `/predict` on 13 leak-free features; label-derived columns
  (`alive`, `class`, `who`, `embark_town`, `alone`) rejected with 422.
- Hermetic pytest suite (preprocessing invariants + serving contract with
  stand-in pipeline fixture); ruff lint; GitHub Actions CI with test gate.
- Dockerized serving image (`bakr1m/titanic-api`) built from the
  release-pinned artifact (SHA256-verified), parity-checked, smoke-tested.
- Professional repo hygiene: LICENSE, CONTRIBUTING, CHANGELOG, CI workflow,
  Makefile, model card, example passenger payload.

### Fixed
- Serving schema previously required the leaky label-derived columns the
  model was trained without; schema now matches the fitted transformer.
- Tests previously required `data/titanic_clean.csv`; synthetic same-schema
  fallback makes them pass on clean checkouts.
- Model previously loaded at import time with a CWD-relative path; now lazy
  `get_pipeline()` behind a repo-root-resolved path (503 if artifact missing).
