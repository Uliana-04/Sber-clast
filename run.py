"""Run directly from the checkout without installing the package."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from municipal_economy.__main__ import main

if __name__ == "__main__":
    main()
