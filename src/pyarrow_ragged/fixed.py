import json

import numpy as np
import pyarrow as pa


class FixedShapeTensorArray(pa.ExtensionArray):
    """Arrow ExtensionArray storing fixed-shape tensors.

    Every element has the same ``shape``. The underlying storage is a
    ``pa.FixedSizeListArray`` over a single flat typed buffer, so the entire
    batch maps to a contiguous array and ``to_numpy`` is a zero-copy reshape.
    """

    @classmethod
    def from_numpy(
        cls, tensors: np.ndarray | list[np.ndarray]
    ) -> "FixedShapeTensorArray":
        """Create a ``FixedShapeTensorArray`` from a NumPy array or a list of arrays.

        Parameters
        ----------
        tensors
            Either a single array of shape ``(N, *shape)`` or a list of arrays
            all sharing the same shape. Non C-contiguous inputs are made
            contiguous before stacking.

        Returns
        -------
        FixedShapeTensorArray
            An ExtensionArray wrapping the provided tensors.
        """
        if isinstance(tensors, np.ndarray):
            arr = np.ascontiguousarray(tensors)
        else:
            arr = np.stack([np.ascontiguousarray(t) for t in tensors])

        shape = arr.shape[1:]
        flat_size = int(np.prod(shape)) if shape else 1

        ext_type = FixedShapeTensorType(shape, arr.dtype)
        storage = pa.FixedSizeListArray.from_arrays(pa.array(arr.ravel()), flat_size)
        return pa.ExtensionArray.from_storage(ext_type, storage)

    def to_numpy(self) -> np.ndarray:
        """Return the contents as a single NumPy array.

        Returns
        -------
        np.ndarray
            Array of shape ``(N, *shape)`` where ``N`` is the number of
            elements. Shares memory with the underlying Arrow buffer where
            possible (zero-copy reshape).
        """
        shape = self.type.shape
        numpy_dtype = self.type.numpy_dtype
        flat = self.storage.values.to_numpy(zero_copy_only=False)
        return flat.astype(numpy_dtype, copy=False).reshape(len(self), *shape)


class FixedShapeTensorType(pa.ExtensionType):
    """Arrow ExtensionType for fixed-shape tensors.

    The element ``shape`` and originating numpy dtype are stored in the
    serialized metadata so the type round-trips correctly through IPC and
    Parquet without any extra configuration.

    Importing ``pyarrow_ragged`` registers this type with PyArrow
    automatically; no explicit registration is required.
    """

    _EXTENSION_NAME = "pyarrow_ragged.fixed_shape_tensor"

    def __init__(
        self,
        shape: tuple = (),
        numpy_dtype: np.dtype = np.dtype("float32"),
    ):
        """
        Parameters
        ----------
        shape
            Fixed dimensions of each element (e.g. ``(4, 5)`` for a batch of
            4×5 matrices). Use ``()`` for scalar elements.
        numpy_dtype
            NumPy dtype of the stored values. Preserved through serialization
            because the Arrow type system does not distinguish all NumPy dtype
            variants (e.g. ``"<U1"`` vs ``"object"``).
        """
        self._shape = tuple(shape)
        self._numpy_dtype = np.dtype(numpy_dtype)
        flat_size = int(np.prod(self._shape)) if self._shape else 1
        super().__init__(
            pa.list_(pa.from_numpy_dtype(self._numpy_dtype), flat_size),
            self._EXTENSION_NAME,
        )

    @property
    def shape(self) -> tuple:
        return self._shape

    @property
    def numpy_dtype(self) -> np.dtype:
        return self._numpy_dtype

    # ------------------------------------------------------------------
    # PyArrow extension-type protocol
    # ------------------------------------------------------------------

    def __arrow_ext_class__(self):
        return FixedShapeTensorArray

    def __arrow_ext_serialize__(self) -> bytes:
        meta = {
            "shape": list(self._shape),
            "numpy_dtype": self._numpy_dtype.str,
        }
        return json.dumps(meta).encode()

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized: bytes):
        meta = json.loads(serialized.decode())
        return cls(tuple(meta["shape"]), np.dtype(meta["numpy_dtype"]))
