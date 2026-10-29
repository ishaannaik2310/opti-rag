"""
OptiRAG-lite Detached Background Worker: SMALL TIER (Groq Allam 2 7B).
Executes Train split (200 questions) followed by Test split (100 questions).
All stdout and stderr redirected to results/log_small.txt.
"""

import sys
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
LOG_FILE = BASE_DIR / "results" / "log_small.txt"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

with open(LOG_FILE, "a", encoding="utf-8") as out:
    out.write(f"\n{'='*70}\n[WORKER SMALL] Starting Small Tier: Train Split\n{'='*70}\n")
    out.flush()

    p_train = subprocess.run(
        [sys.executable, "-u", "run_labels.py", "--tier", "small", "--split", "train", "--live"],
        stdout=out,
        stderr=subprocess.STDOUT,
        cwd=str(BASE_DIR),
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    if p_train.returncode != 0:
        out.write(f"\n[WORKER SMALL] Train split failed with exit code {p_train.returncode}\n")
        out.flush()
        sys.exit(p_train.returncode)

    out.write(f"\n{'='*70}\n[WORKER SMALL] Train split completed. Starting Small Tier: Test Split\n{'='*70}\n")
    out.flush()

    p_test = subprocess.run(
        [sys.executable, "-u", "run_labels.py", "--tier", "small", "--split", "test", "--live"],
        stdout=out,
        stderr=subprocess.STDOUT,
        cwd=str(BASE_DIR),
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    if p_test.returncode != 0:
        out.write(f"\n[WORKER SMALL] Test split failed with exit code {p_test.returncode}\n")
        out.flush()
        sys.exit(p_test.returncode)

    out.write(f"\n{'='*70}\n[WORKER SMALL] ALL DONE (Train + Test finished successfully)\n{'='*70}\n")
    out.flush()

sys.exit(0)
