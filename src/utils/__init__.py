"""
Utility functions for TNShap.
"""

# Add repository root to Python path for imports
import sys
import os
from pathlib import Path

# Get the repository root (3 levels up from this file)
REPO_ROOT = Path(__file__).parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

__all__ = ["REPO_ROOT"]
