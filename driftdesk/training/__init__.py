"""DriftDesk training subpackage — GRPO + SFT pipeline."""
from __future__ import annotations

import csv
import os


def _patch_trainer_log(trainer, csv_path: str) -> None:
    """Monkey-patch trainer.log to also write rows to a CSV file."""
    _written = [os.path.exists(csv_path) and os.path.getsize(csv_path) > 0]
    _orig_log = trainer.log

    def _log_to_csv(logs, *args, **kwargs):
        _orig_log(logs, *args, **kwargs)
        with open(csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=logs.keys())
            if not _written[0]:
                w.writeheader()
                _written[0] = True
            w.writerow(logs)

    trainer.log = _log_to_csv
