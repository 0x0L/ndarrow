"""Ragged and fixed-shape tensor ExtensionArrays and ExtensionTypes for pyarrow.

This module provides lightweight Arrow extension types for storing tensors:

- ``FixedShapeTensorArray`` / ``FixedShapeTensorType``: tensors where every
  element has exactly the same ``shape``.
- ``RaggedTensorArray`` / ``RaggedTensorType``: tensors whose leading dimension
  varies per element (ragged), with a fixed ``inner_shape``.

Both types store the element shape and numpy dtype in the extension metadata so
they round-trip correctly through IPC and Parquet without extra configuration.
"""

import pyarrow as _pa

from .fixed import FixedShapeTensorArray, FixedShapeTensorType
from .ragged import RaggedTensorArray, RaggedTensorType


def _register() -> None:
    """Register extension types with the Arrow type registry.

    PyArrow requires a concrete instance to register, but only uses it
    to map the extension name to the class. The instance's field values
    are irrelevant — every actual type is reconstructed via
    ``__arrow_ext_deserialize__`` at read time.
    """
    for ext_type in (RaggedTensorType, FixedShapeTensorType):
        try:
            _pa.unregister_extension_type(ext_type._EXTENSION_NAME)
        except _pa.ArrowKeyError:
            pass
        _pa.register_extension_type(ext_type())


_register()

__all__ = [
    "RaggedTensorArray",
    "RaggedTensorType",
    "FixedShapeTensorArray",
    "FixedShapeTensorType",
]
