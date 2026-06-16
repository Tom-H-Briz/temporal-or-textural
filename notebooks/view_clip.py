"""
View an SSv2 clip. Set CLIP_ID and run — opens in QuickTime.
"""

import subprocess
from pathlib import Path

ROOT = Path(__file__).parent.parent

# ── Config ────────────────────────────────────────────────────────────────────
CLIP_ID = "144528"

VIDEO_DIR = ROOT / "data/ssv2/20bn-something-something-v2"
# ─────────────────────────────────────────────────────────────────────────────

clip_path = VIDEO_DIR / f"{CLIP_ID}.webm"

if not clip_path.exists():
    raise FileNotFoundError(clip_path)

print(f"Opening: {clip_path}")
subprocess.run(["open", str(clip_path)], check=True)
