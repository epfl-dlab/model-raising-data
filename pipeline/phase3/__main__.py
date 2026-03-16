"""CLI entry point for phase3 improver loop.

Usage:
    uv run python -m pipeline.phase3 --role judge [--aliases olmo3-7B-think]
    uv run python -m pipeline.phase3 --role generator [--aliases olmo3-7B-think]
"""

from pipeline.phase3.loop import main

if __name__ == "__main__":
    main()
