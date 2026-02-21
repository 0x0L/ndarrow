# pyarrow-ragged

PyArrow extension types for efficiently storing tensors with variable or fixed shapes.

## Overview

`pyarrow-ragged` provides two complementary PyArrow-native storage formats for tensor data:

- **`FixedShapeTensorArray`** — every element has exactly the same `shape`; the entire batch is backed by a single contiguous buffer.
- **`RaggedTensorArray`** — each element is a tensor whose leading dimension may vary, with a fixed `inner_shape` shared by all elements.

Both types store the element shape and NumPy dtype in the Arrow extension metadata, so they round-trip correctly through IPC and Parquet without any extra configuration.

## Installation

```bash
pip install pyarrow-ragged
```

## Usage

### Ragged tensors

```python
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow_ragged import RaggedTensorArray

# Varying first dimension, fixed inner shape (4, 5)
tensors = [
    np.random.randn(3, 4, 5).astype(np.float32),
    np.random.randn(7, 4, 5).astype(np.float32),
    np.random.randn(2, 4, 5).astype(np.float32),
]

# inner_shape and dtype are inferred automatically
ragged = RaggedTensorArray.from_numpy(tensors)

table = pa.table({"embeddings": ragged})
print(table.schema)
# embeddings: extension<pyarrow_ragged.ragged_tensor<RaggedTensorType>>

# Round-trip through Parquet — type metadata is preserved
pq.write_table(table, "data.parquet")
table2 = pq.read_table("data.parquet")

# Convert back to a list of NumPy arrays
recovered = table2.column("embeddings").chunk(0).to_numpy()
```

### Fixed-shape tensors

```python
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow_ragged import FixedShapeTensorArray

# All elements share the same shape — pass a stacked array directly
batch = np.random.randn(100, 4, 5).astype(np.float32)

fixed = FixedShapeTensorArray.from_numpy(batch)

table = pa.table({"embeddings": fixed})
print(table.schema)
# embeddings: extension<pyarrow_ragged.fixed_shape_tensor<FixedShapeTensorType>>

# Round-trip through Parquet — type metadata is preserved
pq.write_table(table, "data.parquet")
table2 = pq.read_table("data.parquet")

# Convert back to a single stacked NumPy array of shape (100, 4, 5)
recovered = table2.column("embeddings").chunk(0).to_numpy()
```
