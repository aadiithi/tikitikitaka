"""Put `src/` on the path so the scripts run from a fresh clone with no install.

`pip install -e .` also works and is what the README recommends; this exists so
that `python scripts/train_head.py` works in a Colab cell immediately after
`git clone`, with no install step to forget.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
