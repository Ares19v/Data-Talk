"""
tests/bench.py
──────────────
Full pipeline latency benchmark (text queries only, excludes STT network).
Alias for analytics/latency.py — call it directly or via this alias.

Usage:
    python tests/bench.py --n 100
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analytics.latency import main
if __name__ == "__main__":
    main()
