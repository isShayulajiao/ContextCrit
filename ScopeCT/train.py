#!/usr/bin/env python3
"""Scope-CT starts from the original checkpoint, not the SFT output."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ScopeSFT.launcher import launch

if __name__ == "__main__":
    launch("ct")
