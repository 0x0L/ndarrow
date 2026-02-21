# pyarrow-ragged

A PyArrow extension type for efficiently storing and manipulating ragged tensors—arrays with variable leading dimensions but fixed trailing shapes.

## Overview

`pyarrow-ragged` provides a PyArrow-native storage format for ragged tensor data. Each element in the array is a multidimensional tensor where the outermost dimension may vary while all inner dimensions remain constant. The `inner_shape` and NumPy dtype are stored in the Arrow extension metadata, so the type round-trips correctly through IPC and Parquet without any extra configuration.

## Installation

```bash
pip install pyarrow-ragged
```

## Usage

```python
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow_ragged import RaggedTensorArray

# Create ragged tensors: varying first dimension, fixed inner shape (4, 5)
tensors = [
    np.random.randn(3, 4, 5).astype(np.float32),
    np.random.randn(7, 4, 5).astype(np.float32),
    np.random.randn(2, 4, 5).astype(np.float32),
]

# Build a RaggedTensorArray — inner_shape and dtype are inferred automatically
ragged = RaggedTensorArray.from_numpy(tensors)

# Use it in a PyArrow table
table = pa.table({"embeddings": ragged})
print(table.schema)
# embeddings: extension<pyarrow_ragged.ragged_tensor<LargeListType>>

# Round-trip through Parquet — type metadata is preserved
pq.write_table(table, "data.parquet")
table2 = pq.read_table("data.parquet")

# Convert back to a list of NumPy arrays
recovered = table2.column("embeddings").chunk(0).to_numpy()
```
