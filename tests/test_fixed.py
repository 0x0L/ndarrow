import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ndarrow import FixedShapeTensorArray, FixedShapeTensorType

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def batch_3d():
    rng = np.random.default_rng(0)
    return rng.random((5, 4, 3)).astype(np.float32)


@pytest.fixture
def batch_1d():
    return np.arange(12, dtype=np.float64).reshape(4, 3)


# ---------------------------------------------------------------------------
# FixedShapeTensorType
# ---------------------------------------------------------------------------


class TestFixedShapeTensorType:
    def test_shape(self):
        t = FixedShapeTensorType((4, 3))
        assert t.shape == (4, 3)

    def test_numpy_dtype_default(self):
        t = FixedShapeTensorType(())
        assert t.numpy_dtype == np.dtype("float32")

    def test_numpy_dtype_custom(self):
        t = FixedShapeTensorType((3,), np.dtype("float64"))
        assert t.numpy_dtype == np.dtype("float64")

    def test_serialize_roundtrip(self):
        t = FixedShapeTensorType((4, 3), np.dtype("float64"))
        serialized = t.__arrow_ext_serialize__()
        t2 = FixedShapeTensorType.__arrow_ext_deserialize__(t.storage_type, serialized)
        assert t2.shape == (4, 3)
        assert t2.numpy_dtype == np.dtype("float64")

    def test_extension_name(self):
        assert FixedShapeTensorType._EXTENSION_NAME == "ndarrow.fixed_shape_tensor"


# ---------------------------------------------------------------------------
# FixedShapeTensorArray.from_numpy
# ---------------------------------------------------------------------------


class TestFromNumpy:
    def test_from_stacked_array(self, batch_3d):
        arr = FixedShapeTensorArray.from_numpy(batch_3d)
        assert isinstance(arr, FixedShapeTensorArray)
        assert arr.type.shape == (4, 3)

    def test_from_list(self, batch_3d):
        tensors = list(batch_3d)
        arr = FixedShapeTensorArray.from_numpy(tensors)
        assert arr.type.shape == (4, 3)

    def test_dtype_inferred(self, batch_3d):
        arr = FixedShapeTensorArray.from_numpy(batch_3d)
        assert arr.type.numpy_dtype == np.dtype("float32")

    def test_length(self, batch_3d):
        arr = FixedShapeTensorArray.from_numpy(batch_3d)
        assert len(arr) == len(batch_3d)

    def test_fortran_order(self):
        t = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32, order="F")
        arr = FixedShapeTensorArray.from_numpy(t)
        assert np.array_equal(arr.to_numpy(), t)

    def test_non_contiguous_slice(self):
        base = np.arange(30, dtype=np.float32).reshape(5, 6)
        t = base[:, ::2]  # non-contiguous
        arr = FixedShapeTensorArray.from_numpy(t)
        assert np.array_equal(arr.to_numpy(), t)


# ---------------------------------------------------------------------------
# FixedShapeTensorArray.to_numpy
# ---------------------------------------------------------------------------


class TestToNumpy:
    def test_roundtrip_3d(self, batch_3d):
        arr = FixedShapeTensorArray.from_numpy(batch_3d)
        assert np.allclose(arr.to_numpy(), batch_3d)

    def test_roundtrip_1d(self, batch_1d):
        arr = FixedShapeTensorArray.from_numpy(batch_1d)
        assert np.allclose(arr.to_numpy(), batch_1d)

    def test_returns_single_array(self, batch_3d):
        arr = FixedShapeTensorArray.from_numpy(batch_3d)
        result = arr.to_numpy()
        assert isinstance(result, np.ndarray)
        assert result.shape == batch_3d.shape

    def test_dtype_preserved_float64(self, batch_1d):
        arr = FixedShapeTensorArray.from_numpy(batch_1d)
        assert arr.to_numpy().dtype == np.dtype("float64")

    def test_dtype_preserved_string(self):
        batch = np.array([["a", "b"], ["c", "d"]])
        arr = FixedShapeTensorArray.from_numpy(batch)
        result = arr.to_numpy()
        assert result.dtype == np.dtype("<U1")
        assert np.array_equal(result, batch)


# ---------------------------------------------------------------------------
# PyArrow table integration
# ---------------------------------------------------------------------------


class TestTableIntegration:
    def test_schema_type(self, batch_3d):
        arr = FixedShapeTensorArray.from_numpy(batch_3d)
        tbl = pa.table({"tensor": arr})
        assert isinstance(tbl.schema.field("tensor").type, FixedShapeTensorType)

    def test_shape_in_schema(self, batch_3d):
        arr = FixedShapeTensorArray.from_numpy(batch_3d)
        tbl = pa.table({"tensor": arr})
        assert tbl.schema.field("tensor").type.shape == (4, 3)

    def test_chunk_class(self, batch_3d):
        arr = FixedShapeTensorArray.from_numpy(batch_3d)
        tbl = pa.table({"tensor": arr})
        assert isinstance(tbl.column("tensor").chunk(0), FixedShapeTensorArray)


# ---------------------------------------------------------------------------
# Parquet round-trip
# ---------------------------------------------------------------------------


class TestParquet:
    def test_type_survives_parquet(self, batch_3d, tmp_path):
        arr = FixedShapeTensorArray.from_numpy(batch_3d)
        tbl = pa.table({"tensor": arr})
        path = tmp_path / "data.parquet"
        pq.write_table(tbl, path)
        tbl2 = pq.read_table(path)
        field = tbl2.schema.field("tensor")
        assert isinstance(field.type, FixedShapeTensorType)
        assert field.type.shape == (4, 3)
        assert field.type.numpy_dtype == np.dtype("float32")

    def test_values_survive_parquet(self, batch_3d, tmp_path):
        arr = FixedShapeTensorArray.from_numpy(batch_3d)
        tbl = pa.table({"tensor": arr})
        path = tmp_path / "data.parquet"
        pq.write_table(tbl, path)
        recovered = pq.read_table(path).column("tensor").chunk(0).to_numpy()
        assert np.allclose(recovered, batch_3d)
