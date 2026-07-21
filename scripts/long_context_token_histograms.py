#!/usr/bin/env python3
"""Report prefix lengths for the local long-context dataset by family and format."""

from __future__ import annotations

import sys

from state_bench.token_histogram import main

if __name__ == "__main__":
    raise SystemExit(
        main(
            [
                "data/state-tracking-long-context-v1/rendered",
                "--group-by",
                "formatter",
                "complexity",
                *sys.argv[1:],
            ]
        )
    )
