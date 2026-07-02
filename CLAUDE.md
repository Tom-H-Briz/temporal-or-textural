# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Coding Commandments
Do not write any more than 20 lines at a time. 
All code changes must be accompanied by a bullet point rationale. 

## Package Management

This project uses `uv`. Always use `uv` rather than `pip` or `python` directly.

```bash
uv sync                  # install / sync dependencies
uv run python <script>   # run a script in the venv
uv add <package>         # add a dependency
```

Python 3.12 is pinned in `.python-version`.

## Project Overview

**temporal-or-textural** is a mechanistic interpretability research project investigating whether video understanding models learn temporal reasoning vs. textural shortcuts. It uses the Something-Something v2 (SSv2) dataset, which is specifically designed to require temporal understanding (actions that cannot be identified from a single frame).

## Coding Conventions

Style: functional Python — no unnecessary classes
Config: config dicts at the top of each script, not hardcoded constants scattered through functions
Function length: 30–50 lines; break up anything longer
use simple code, i'm stupic.  
Hookpoints: parameterised by layer index — never hardcoded to VideoMAE internals (model-agnostic by design)
Precision: float16 for activations, float32 for SAE weights
Do not write any more than 20 lines at a time. 
All code changes must be accompanied by a bullet point rationale. 

## Key Dependencies

torch / torchvision — model inference and SAE training
transformers / huggingface-hub — VideoMAE loading
wandb — experiment tracking
pandas / pyarrow — parquet files for metadata and results
tqdm — progress bars
Overcomplete library fucntions that have been 

## Data & Artifacts
data/, models/, outputs/, *.pt, and *.parquet are all gitignored and live only locally.
Separation of Concerns
Conceptual and research design work happens in a separate Claude chat instance. This Claude Code instance is for implementation only. Do not propose changes to experimental design — implement what is specified.