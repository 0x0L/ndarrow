import json

import numpy as np
import pyarrow as pa


class RaggedTensorArray(pa.ExtensionArray):
    """Arrow ExtensionArray storing ragged tensors.

    Each element is a 0..N row tensor with a fixed ``inner_shape``. The
    underlying storage is a ``pa.ListArray`` over a single flat buffer.
    """

    @classmethod
    def from_numpy(cls, tensors: list[np.ndarray]) -> RaggedTensorArray:
        """Create a `RaggedTensorArray` from a sequence of numpy arrays.

        Parameters
        ----------
        tensors
            Sequence of numpy arrays. All arrays must share the same
            trailing shape (``inner_shape``) and may differ in the leading
            dimension.

        Returns
        -------
        RaggedTensorArray
            An ExtensionArray wrapping the provided tensors.

        Notes
        -----
        The function does not copy C-contiguous arrays; non-contiguous
        inputs will be made contiguous as needed.
        """

        c_tensor = np.concatenate([np.ascontiguousarray(t) for t in tensors])
        inner_shape = c_tensor.shape[1:]
        flat = c_tensor.ravel()
        offsets = np.zeros(len(tensors) + 1, dtype=np.int64)
        for i, t in enumerate(tensors):
            offsets[i + 1] = offsets[i] + t.size

        ext_type = RaggedTensorType(inner_shape, flat.dtype)
        storage = pa.LargeListArray.from_arrays(pa.array(offsets), pa.array(flat))
        return pa.ExtensionArray.from_storage(ext_type, storage)

    def to_numpy(self) -> list[np.ndarray]:
        """Return the contents as a list of numpy arrays.

        The returned arrays are views into a single reshaped buffer where
        possible (no per-element copy is performed beyond dtype casting
        when needed).
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

    _EXTENSION_NAME = "pyarrow_ragged.ragged_tensor"

    def __init__(
        self,
        inner_shape: tuple = tuple(),
        numpy_dtype: np.dtype = np.dtype("float32"),
    ):
        self._inner_shape = tuple(inner_shape)
        self._numpy_dtype = np.dtype(numpy_dtype)
        super().__init__(
            pa.large_list(pa.from_numpy_dtype(self._numpy_dtype)),
            self._EXTENSION_NAME,
        )

    @property
    def inner_shape(self) -> tuple:
        return self._inner_shape

    @property
    def numpy_dtype(self) -> np.dtype:
        return self._numpy_dtype

    def __arrow_ext_class__(self):
        return RaggedTensorArray

    def __arrow_ext_serialize__(self) -> bytes:
        meta = {
            "inner_shape": list(self._inner_shape),
            "numpy_dtype": self._numpy_dtype.str,  # e.g. "<f4", "<U1"
        }
        return json.dumps(meta).encode()

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized: bytes):
        # numpy_dtype is stored in the serialized bytes because the Arrow
        # type system does not preserve all numpy dtype distinctions (e.g. "<U1").
        meta = json.loads(serialized.decode())
        return cls(tuple(meta["inner_shape"]), np.dtype(meta["numpy_dtype"]))

    @classmethod
    def register(cls) -> None:
        """Register this extension type with the Arrow type registry.

        PyArrow requires a concrete instance to register, but only uses it
        to map the extension name to the class. The instance's field values
        (inner_shape, numpy_dtype) are irrelevant — every actual type is
        reconstructed via ``__arrow_ext_deserialize__`` at read time.
        """
        try:
            pa.unregister_extension_type(cls._EXTENSION_NAME)
        except pa.ArrowKeyError:
            pass
        pa.register_extension_type(cls())
