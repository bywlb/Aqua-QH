"""One-command offline integration test: does not download model weights."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
cmd = [sys.executable, "-m", "fish_qhprp.run_main",
       "--input", "examples/synthetic_candidates.json",
       "--output-dir", "outputs/smoke",
       "--qrels", "examples/synthetic_qrels.txt",
       "--judge", "mock", "--encoder", "mock", "--num-topics", "3"]
print("Running:", " ".join(cmd), flush=True)
subprocess.run(cmd, check=True, cwd=ROOT)
