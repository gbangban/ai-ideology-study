#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.teacher.build_grpo_dataset import main as build_main
if __name__ == "__main__":
    build_main()
