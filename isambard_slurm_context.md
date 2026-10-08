# Isambard-AI Slurm Context for Claude Code

## System

- **Machine**: Isambard-AI Phase 2
- **CPU architecture**: `aarch64` (ARM) — important for package compatibility
- **GPUs**: Nvidia GH200 Superchips — 1 GPU = 1 complete GH200
- **Scheduler**: Slurm
- **Max walltime**: 24 hours — shorter walltimes get better queue priority

---

## Key Paths

| Variable | Path | Use for |
|---|---|---|
| `$HOME` | `/home/b6o/tomheslin83.b6o` | Repo, scripts, venv, job outputs |
| `$SCRATCHDIR` | `/scratch/b6o/tomheslin83.b6o` | Input data, videos |
| `$PROJECTDIR` | `/projects/b6o` | Shared datasets, shared environments |

> ⚠️ Nothing is backed up. `$SCRATCHDIR` files not accessed for 60 days are deleted.

---

## Repo Location

The experiment repo is cloned at:
```
/home/b6o/tomheslin83.b6o/<reponame>/
```

The Slurm script should live inside the repo and be submitted from `$HOME/<reponame>/`.

---

## Video Data Location

```
/scratch/b6o/tomheslin83.b6o/videos/
```

---

## Environment — uv

- `uv` is available system-wide (version 0.9.7)
- Environment must be set up **inside the job** — do not rely on login node setup
- `uv venv` + `uv sync` is the correct pattern
- The venv should be created at `$HOME/<reponame>/`
- **Do not** `uv venv` on the login node — GPU packages fail without a GPU present

### pyproject.toml structure

`[tool.uv.sources]` must come **after** the `[project]` block with `dependencies` inside it:

```toml
[project]
name = "your-project"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "torch>=2.12.0",
    "torchvision>=0.27.0",
    # ... other deps
]

[tool.uv.sources]
torch = { index = "pytorch-cu128" }
torchvision = { index = "pytorch-cu128" }
torchaudio = { index = "pytorch-cu128" }

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true
```

---

## Environment Variables

```bash
export HF_TOKEN="your_token_here"   # HuggingFace — required for model downloads
```

---

## Slurm Script Template

```bash
#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --output=my_job.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00

cd $HOME/<reponame>
uv venv
source .venv/bin/activate
uv sync

export HF_TOKEN="your_token_here"

uv run python your_script.py
```

---

## Output Paths

Scripts must write outputs relative to their own location — not parent directories:

```python
# Correct
OUTPUT_DIR = Path(__file__).parent / "outputs"

# Wrong — will hit permission error
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
```

---

## Job Commands

```bash
# Submit
sbatch my_job.sh

# Monitor (PD = pending, R = running)
squeue --me

# Cancel
scancel <JOBID>

# View output
cat my_job.out
```

---

## Useful Links

- Docs: https://docs.isambard.ac.uk
- Portal: https://portal.isambard.ac.uk
- Support: https://support.isambard.ac.uk
- Status: https://status.isambard.ac.uk
