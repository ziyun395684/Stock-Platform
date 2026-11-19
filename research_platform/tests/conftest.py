from __future__ import annotations

import os

# Keep API contract tests isolated from the developer's MySQL database.
os.environ["STOCKRNN_STORAGE_BACKEND"] = "memory"
