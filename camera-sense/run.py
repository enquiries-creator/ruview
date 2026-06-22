#!/usr/bin/env python3
"""Entry point: python run.py --config config.yaml"""

from __future__ import annotations

import argparse
import logging
import sys

from camera_sense.app import Application
from camera_sense.config import load_config


def main() -> int:
    ap = argparse.ArgumentParser(description="camera_sense — camera presence/approach sensing → Home Assistant")
    ap.add_argument("--config", "-c", default="config.yaml", help="path to config YAML")
    args = ap.parse_args()

    cfg = load_config(args.config)
    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    Application(cfg).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
