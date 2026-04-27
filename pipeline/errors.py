"""
Two error categories the pipeline distinguishes:

- RecoverableError: transient — propagate so Modal's `retries=` policy
  re-spawns the function. The next attempt picks up from the latest
  committed checkpoint.

- FatalError: deterministic and won't fix itself (bad input, schema
  validation failed, etc.). Caught at the top of train_pipeline,
  marks the job dead, and skips the retry loop.

Anything else (raw Exception) is treated as recoverable by Modal's
retry policy. Convert to FatalError early if you know it's hopeless.
"""

from __future__ import annotations


class RecoverableError(Exception):
    """Transient — let Modal retry."""


class FatalError(Exception):
    """Permanent — mark job dead, don't retry."""
