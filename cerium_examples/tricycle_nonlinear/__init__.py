import sys
from pathlib import Path

try:
    import tricycle  # noqa: F401
except ImportError:
    # Copied to Cerium/examples/tricycle_nonlinear inside Cerium-With-Tricycle.
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
