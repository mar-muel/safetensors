# Re-export this
from ._safetensors_rust import (  # noqa: F401
    SafetensorError,
    __version__,
    deserialize,
    safe_open,
    safe_write,
    serialize,
    serialize_file,
)
