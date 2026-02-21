import json
from collections.abc import Sequence

import numpy as np
import pyarrow as pa


class RaggedTensorArray(pa.ExtensionArray):
    """Arrow ExtensionArray storing ragged tensors.

    Each element is a 0..N row tensor with a fixed ``inner_shape``. The
    underlying storage is a ``pa.LargeListArray`` over a single flat buffer.
    """

    @classmethod
    def from_numpy(cls, tensors: Sequence[np.ndarray]) -> "RaggedTensorArray":
        """Create a `RaggedTensorArray` from a sequence of numpy arrays.

        Parameters
        ----------
        tensors
            All arrays must share the same trailing shape (``inner_shape``)
            and may differ only in the leading dimension.

        Returns
        -------
        RaggedTensorArray
            An ExtensionArray wrapping the provided tensors.

        Raises
        ------
        ValueError
            If ``tensors`` is empty or arrays have mismatched trailing shapes.

        Notes
        -----
        A new buffer is always allocated by the concatenation step.
        Non C-contiguous inputs incur an additional copy to make them
        C-contiguous before concatenation.
        """
        if len(tensors) == 0:
            raise ValueError("tensors must not be empty")

        inner_shape = tensors[0].shape[1:]
        for i, t in enumerate(tensors[1:], 1):
            if t.shape[1:] != inner_shape:
                raise ValueError(
                    f"All tensors must share the same trailing shape (inner_shape). "
                    f"Expected {inner_shape!r} (inferred from index 0) "
                    f"but got {t.shape[1:]!r} at index {i}."
                )

        c_tensor = np.concatenate([np.ascontiguousarray(t) for t in tensors])
        flat = c_tensor.ravel()

        offsets = np.zeros(len(tensors) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum([t.size for t in tensors])

        ext_type = RaggedTensorType(inner_shape, flat.dtype)
        storage = pa.LargeListArray.from_arrays(pa.array(offsets), pa.array(flat))
        return pa.ExtensionArray.from_storage(ext_type, storage)

    def to_numpy(self) -> list[np.ndarray]:
        """Return the contents as a list of numpy arrays.

        Returns
        -------
        list[np.ndarray]
            One array per element, each with shape ``(n, *inner_shape)`` where
            ``n`` is the leading dimension of that element.

        Notes
        -----
        The returned arrays are views into a single reshaped buffer where
        possible; no per-element copy is performed beyond dtype casting
        when needed.
        """
        inner_shape = self.type.inner_shape
        numpy_dtype = self.type.numpy_dtype
        inner_size = int(np.prod(inner_shape)) if inner_shape else 1

        flat = self.storage.values.to_numpy(zero_copy_only=False)
        offsets = self.storage.offsets.to_numpy(zero_copy_only=False)

        flat_reshaped = flat.astype(numpy_dtype, copy=False).reshape(-1, *inner_shape)
        row_offsets = offsets // inner_size

        return [
            flat_reshaped[row_offsets[i] : row_offsets[i + 1]]
            for i in range(len(row_offsets) - 1)
        ]


class RaggedTensorType(pa.ExtensionType):
    """Arrow ExtensionType for ragged tensors.

    The extension stores the fixed trailing ``inner_shape`` and the
    originating numpy dtype in the serialized metadata so that the type
    can be reconstructed exactly when deserialized.
    """

    _EXTENSION_NAME = "ndarrow.ragged_tensor"

    def __init__(
        self,
        inner_shape: tuple[int, ...] = (),
        numpy_dtype: np.dtype = np.dtype("float32"),
    ):
        """
        Parameters
        ----------
        inner_shape
            Fixed trailing dimensions shared by every element (e.g. ``(4, 5)``
            for a batch of 4×5 matrices). Use ``()`` for 1-D variable-length
            arrays.
        numpy_dtype
            NumPy dtype of the stored values. Preserved through
            serialization because the Arrow type system does not distinguish
            all NumPy dtype variants (e.g. ``"<U1"`` vs ``"object"``).
        """
        self._inner_shape = tuple(inner_shape)
        self._numpy_dtype = np.dtype(numpy_dtype)
        super().__init__(
            pa.large_list(pa.from_numpy_dtype(self._numpy_dtype)),
            self._EXTENSION_NAME,
        )

    @property
    def inner_shape(self) -> tuple[int, ...]:
        return self._inner_shape

    @property
    def numpy_dtype(self) -> np.dtype:
        return self._numpy_dtype

    def __repr__(self) -> str:
        return f"RaggedTensorType(inner_shape={self._inner_shape}, dtype={self._numpy_dtype})"

    def equals(self, other: object) -> bool:
        """Return True if *other* is the same type with the same parameters.

        Use this instead of ``==`` for semantic comparison. PyArrow's
        ``pa.ExtensionType`` compares using only the underlying storage type
        (a C-level slot), so two ``RaggedTensorType`` instances with different
        ``inner_shape`` but the same element dtype would incorrectly compare
        as equal with ``==``.
        """
        return (
            isinstance(other, RaggedTensorType)
            and self._inner_shape == other._inner_shape
            and self._numpy_dtype == other._numpy_dtype
        )

    # ------------------------------------------------------------------
    # PyArrow extension-type protocol
    # ------------------------------------------------------------------

    def __arrow_ext_class__(self) -> type[RaggedTensorArray]:
        return RaggedTensorArray

    def __arrow_ext_serialize__(self) -> bytes:
        meta = {
            "inner_shape": list(self._inner_shape),
            "numpy_dtype": self._numpy_dtype.str,  # e.g. "<f4", "<U1"
        }
        return json.dumps(meta).encode()

    @classmethod
    def __arrow_ext_deserialize__(
        cls, storage_type, serialized: bytes
    ) -> "RaggedTensorType":
        # numpy_dtype is stored in the serialized bytes because the Arrow
        # type system does not preserve all numpy dtype distinctions (e.g. "<U1").
        meta = json.loads(serialized.decode())
        return cls(tuple(meta["inner_shape"]), np.dtype(meta["numpy_dtype"]))
