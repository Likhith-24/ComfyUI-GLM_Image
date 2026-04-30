"""GLM-Image custom nodes — separate-loader architecture only.

Legacy monolithic pipeline loader and pipeline-extract nodes were removed
on 2026-05-01 per user request ("Remove legacy. Add I2I support.").
"""

from .separate_nodes import (
    SEPARATE_NODE_CLASS_MAPPINGS,
    SEPARATE_NODE_DISPLAY_NAME_MAPPINGS,
)

NODE_CLASS_MAPPINGS = dict(SEPARATE_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS = dict(SEPARATE_NODE_DISPLAY_NAME_MAPPINGS)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
