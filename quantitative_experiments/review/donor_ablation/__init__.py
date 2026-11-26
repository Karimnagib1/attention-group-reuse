"""Isolated donor-robustness experiments; do not write bytecode into reused code."""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
QUANT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(QUANT_ROOT / "src"))
