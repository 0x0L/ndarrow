# ndarrow

PyArrow extension types for efficiently storing tensors with variable or fixed shapes.

## Overview

`ndarrow` provides two complementary PyArrow-native storage formats for tensor data:

- **`TensorArray`** — every element has exactly the same `shape`; the entire batch is backed by a single contiguous buffer.
- **`RaggedTensorArray`** — each element is a tensor whose leading dimension may vary, with a fixed `inner_shape` shared by all elements.

Both types store the element shape and NumPy dtype in the Arrow extension metadata, so they round-trip correctly through IPC and Parquet without any extra configuration.

## Installation

```bash
pip install ndarrow
```

## Usage

```python
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from ndarrow import TensorArray, RaggedTensorArray

# Fixed-shape: one 64-dim embedding per sentence
sentence_embeddings = TensorArray.from_numpy(np.random.randn(3, 64))
# also accepts a list of arrays:
# sentence_embeddings = TensorArray.from_numpy([
#     np.random.randn(64)
#     for _ in range(3)
# ])

# Ragged: each sentence has a variable number of tokens, each with a 64-dim embedding
token_embeddings = RaggedTensorArray.from_numpy([
    np.random.randn(6, 64),
    np.random.randn(9, 64),
    np.random.randn(3, 64),
])

table = pa.table({"sentence_embeddings": sentence_embeddings, "token_embeddings": token_embeddings})
print(table.schema)
# sentence_embeddings: extension<ndarrow.tensor<TensorType>>
# token_embeddings:    extension<ndarrow.ragged_tensor<RaggedTensorType>>

# Round-trip through Parquet — type metadata is preserved
pq.write_table(table, "data.parquet")
table2 = pq.read_table("data.parquet")

embeddings_np = table2.column("sentence_embeddings").chunk(0).to_numpy()  # shape (3, 64)
tokens_list   = table2.column("token_embeddings").chunk(0).to_numpy()     # list of 3 arrays of shape (?, 64)
```
