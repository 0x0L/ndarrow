import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ndarrow import RaggedTensorArray, RaggedTensorType

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tensors_3d():
    rng = np.random.default_rng(0)
    return [
        rng.random((3, 4, 5)).astype(np.float32),
        rng.random((7, 4, 5)).astype(np.float32),
        rng.random((2, 4, 5)).astype(np.float32),
    ]


@pytest.fixture
def tensors_1d():
    return [
        np.array([1.0, 2.0, 3.0], dtype=np.float32),
        np.array([4.0, 5.0], dtype=np.float32),
        np.array([6.0, 7.0, 8.0, 9.0], dtype=np.float32),
    ]


# ---------------------------------------------------------------------------
# RaggedTensorType
# ---------------------------------------------------------------------------


class TestRaggedTensorType:
    def test_inner_shape(self):
        t = RaggedTensorType((4, 5))
        assert t.inner_shape == (4, 5)

    def test_numpy_dtype_default(self):
        t = RaggedTensorType(())
        assert t.numpy_dtype == np.dtype("float32")

    def test_numpy_dtype_custom(self):
        t = RaggedTensorType((3,), np.dtype("float64"))
        assert t.numpy_dtype == np.dtype("float64")

    def test_serialize_roundtrip(self):
        t = RaggedTensorType((4, 5), np.dtype("float64"))
        serialized = t.__arrow_ext_serialize__()
        t2 = RaggedTensorType.__arrow_ext_deserialize__(t.storage_type, serialized)
        assert t2.inner_shape == (4, 5)
        assert t2.numpy_dtype == np.dtype("float64")

    def test_extension_name(self):
        assert RaggedTensorType._EXTENSION_NAME == "ndarrow.ragged_tensor"


# ---------------------------------------------------------------------------
# RaggedTensorArray.from_numpy
# ---------------------------------------------------------------------------


class TestFromNumpy:
    def test_inner_shape_inferred(self, tensors_3d):
        arr = RaggedTensorArray.from_numpy(tensors_3d)
        assert arr.type.inner_shape == (4, 5)

    def test_dtype_inferred(self, tensors_3d):
        arr = RaggedTensorArray.from_numpy(tensors_3d)
        assert arr.type.numpy_dtype == np.dtype("float32")

    def test_1d(self, tensors_1d):
        arr = RaggedTensorArray.from_numpy(tensors_1d)
        assert arr.type.inner_shape == ()

    def test_mismatched_inner_shape_raises(self):
        with pytest.raises(ValueError):
            RaggedTensorArray.from_numpy(
                [
                    np.ones((3, 4), dtype=np.float32),
                    np.ones((3, 5), dtype=np.float32),
                ]
            )

    def test_fortran_order(self):
        t = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32, order="F")
        arr = RaggedTensorArray.from_numpy([t])
        rec = arr.to_numpy()[0]
        assert np.array_equal(t, rec)

    def test_non_contiguous_slice(self):
        base = np.arange(20, dtype=np.float32).reshape(4, 5)
        t = base[::2]  # non-contiguous
        arr = RaggedTensorArray.from_numpy([t])
        assert np.array_equal(arr.to_numpy()[0], t)

    def test_returns_correct_class(self, tensors_3d):
        arr = RaggedTensorArray.from_numpy(tensors_3d)
        assert isinstance(arr, RaggedTensorArray)


# ---------------------------------------------------------------------------
# RaggedTensorArray.to_numpy
# ---------------------------------------------------------------------------


class TestToNumpy:
    def test_roundtrip_3d(self, tensors_3d):
        arr = RaggedTensorArray.from_numpy(tensors_3d)
        recovered = arr.to_numpy()
        for orig, rec in zip(tensors_3d, recovered):
            assert np.allclose(orig, rec)

    def test_roundtrip_1d(self, tensors_1d):
        arr = RaggedTensorArray.from_numpy(tensors_1d)
        recovered = arr.to_numpy()
        for orig, rec in zip(tensors_1d, recovered):
            assert np.allclose(orig, rec)

    def test_shapes_preserved(self, tensors_3d):
        recovered = RaggedTensorArray.from_numpy(tensors_3d).to_numpy()
        for orig, rec in zip(tensors_3d, recovered):
            assert rec.shape == orig.shape

    def test_dtype_preserved_float64(self):
        tensors = [np.ones((3, 4), dtype=np.float64)]
        rec = RaggedTensorArray.from_numpy(tensors).to_numpy()[0]
        assert rec.dtype == np.dtype("float64")

    def test_dtype_preserved_string(self):
        tensors = [np.array(["a", "b"]), np.array(["c"])]
        rec = RaggedTensorArray.from_numpy(tensors).to_numpy()
        assert rec[0].dtype == np.dtype("<U1")
        assert rec[1].dtype == np.dtype("<U1")

    def test_output_is_c_contiguous(self):
        t = np.ones((3, 4), dtype=np.float32, order="F")
        rec = RaggedTensorArray.from_numpy([t]).to_numpy()[0]
        assert rec.flags["C_CONTIGUOUS"]

    def test_length(self, tensors_3d):
        arr = RaggedTensorArray.from_numpy(tensors_3d)
        assert len(arr.to_numpy()) == len(tensors_3d)


# ---------------------------------------------------------------------------
# PyArrow table integration
# ---------------------------------------------------------------------------


class TestTableIntegration:
    def test_schema_type(self, tensors_3d):
        arr = RaggedTensorArray.from_numpy(tensors_3d)
        tbl = pa.table({"tensor": arr})
        assert isinstance(tbl.schema.field("tensor").type, RaggedTensorType)

    def test_inner_shape_in_schema(self, tensors_3d):
        arr = RaggedTensorArray.from_numpy(tensors_3d)
        tbl = pa.table({"tensor": arr})
        assert tbl.schema.field("tensor").type.inner_shape == (4, 5)

    def test_chunk_class(self, tensors_3d):
        arr = RaggedTensorArray.from_numpy(tensors_3d)
        tbl = pa.table({"tensor": arr})
        assert isinstance(tbl.column("tensor").chunk(0), RaggedTensorArray)


# ---------------------------------------------------------------------------
# Parquet round-trip
# ---------------------------------------------------------------------------


class TestParquet:
    def test_type_survives_parquet(self, tensors_3d, tmp_path):
        arr = RaggedTensorArray.from_numpy(tensors_3d)
        tbl = pa.table({"tensor": arr})
        path = tmp_path / "data.parquet"
        pq.write_table(tbl, path)
        tbl2 = pq.read_table(path)
        field = tbl2.schema.field("tensor")
        assert isinstance(field.type, RaggedTensorType)
        assert field.type.inner_shape == (4, 5)
        assert field.type.numpy_dtype == np.dtype("float32")

    def test_values_survive_parquet(self, tensors_3d, tmp_path):
        arr = RaggedTensorArray.from_numpy(tensors_3d)
        tbl = pa.table({"tensor": arr})
        path = tmp_path / "data.parquet"
        pq.write_table(tbl, path)
        recovered = pq.read_table(path).column("tensor").chunk(0).to_numpy()
        for orig, rec in zip(tensors_3d, recovered):
            assert np.allclose(orig, rec)
