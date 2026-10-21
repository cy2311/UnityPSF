"""Vendored smlm_v2a runtime for Neptune.

This package keeps a compatibility alias under ``smlm_v2a`` so older
vendored modules that still use absolute imports continue to resolve
inside the Neptune tree rather than falling back to the workspace-level
``/race/smlm_v2a`` package.
"""

from __future__ import annotations

import sys


# Register the vendored package as the canonical ``smlm_v2a`` import root
# before submodules are imported.
sys.modules.setdefault("smlm_v2a", sys.modules[__name__])
