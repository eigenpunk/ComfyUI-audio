import folder_paths

from .util_nodes import (
    NODE_CLASS_MAPPINGS as UTIL_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as UTIL_NODE_DISPLAY_NAME_MAPPINGS,
)
from .musicgen_nodes import (
    NODE_CLASS_MAPPINGS as MGAC_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as MGAC_NODE_DISPLAY_NAME_MAPPINGS,
)
from .musicgen_hf_nodes import (
    NODE_CLASS_MAPPINGS as MGHF_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as MGHF_NODE_DISPLAY_NAME_MAPPINGS,
)
from .tortoise_nodes import (
    NODE_CLASS_MAPPINGS as TORTOISE_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as TORTOISE_NODE_DISPLAY_NAME_MAPPINGS,
)
from .valle_x_nodes import (
    NODE_CLASS_MAPPINGS as VEX_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as VEX_NODE_DISPLAY_MAPPINGS,
)


NODE_CLASS_MAPPINGS = {
    **UTIL_NODE_CLASS_MAPPINGS,
    **MGAC_NODE_CLASS_MAPPINGS,
    **MGHF_NODE_CLASS_MAPPINGS,
    **TORTOISE_NODE_CLASS_MAPPINGS,
    **VEX_NODE_CLASS_MAPPINGS,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    **UTIL_NODE_DISPLAY_NAME_MAPPINGS,
    **MGAC_NODE_DISPLAY_NAME_MAPPINGS,
    **MGHF_NODE_DISPLAY_NAME_MAPPINGS,
    **TORTOISE_NODE_DISPLAY_NAME_MAPPINGS,
    **VEX_NODE_DISPLAY_MAPPINGS,
}
