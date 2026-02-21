"""Ragged tensor ExtensionArray and ExtensionType for pyarrow.

This module provides a lightweight Arrow extension type that represents
arrays of tensors whose first dimension (the number of rows) may vary
between elements. The data is stored as a `pa.list_` over a flat
buffer, with the element (inner) shape and numpy dtype stored in the
extension metadata so the type round-trips correctly through IPC/Parquet.
"""

from .ragged import RaggedTensorArray, RaggedTensorType


def _register() -> None:
    """Register this extension type with the Arrow type registry.

    PyArrow requires a concrete instance to register, but only uses it
    to map the extension name to the class. The instance's field values
    (inner_shape, numpy_dtype) are irrelevant — every actual type is
    reconstructed via ``__arrow_ext_deserialize__`` at read time.
    """
    import pyarrow as pa

    try:
        pa.unregister_extension_type(RaggedTensorType._EXTENSION_NAME)
    except pa.ArrowKeyError:
        pass
    pa.register_extension_type(RaggedTensorType())


_register()

__all__ = ["RaggedTensorArray", "RaggedTensorType"]
