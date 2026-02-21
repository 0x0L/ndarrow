"""Ragged tensor ExtensionArray and ExtensionType for pyarrow.

This module provides a lightweight Arrow extension type that represents
arrays of tensors whose first dimension (the number of rows) may vary
between elements. The data is stored as a `pa.list_` over a flat
buffer, with the element (inner) shape and numpy dtype stored in the
extension metadata so the type round-trips correctly through IPC/Parquet.
"""

from .ragged import RaggedTensorArray, RaggedTensorType

RaggedTensorType.register()

__all__ = ["RaggedTensorArray", "RaggedTensorType"]
