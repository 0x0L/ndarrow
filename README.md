# pyarrow-ragged

A PyArrow extension type for efficiently storing and manipulating ragged tensors—arrays with variable leading dimensions but fixed trailing shapes.

## Overview

`pyarrow-ragged` provides a PyArrow-native storage format for ragged tensor data. Each element in the array is a multidimensional tensor where the outermost dimension may vary while all inner dimensions remain constant. This design enables efficient serialization, columnar storage, and interoperability with PyArrow-based systems.

## Installation

```bash
pip install pyarrow-ragged
```

## Usage

### Creating a RaggedTensorArray

```python
import numpy as np
from pyarrow_ragged import RaggedTensorArray

# Create ragged tensors (varying first dimension, fixed inner shape)
tensors = [
    np.random.randn(3, 4, 5).astype(np.float32),
    np.random.randn(7, 4, 5).astype(np.float32),
    np.random.randn(2, 4, 5).astype(np.float32),
]

# Convert to RaggedTensorArray
ragged_array = RaggedTensorArray.from_numpy(tensors)
```

### Converting Back to NumPy

```python
# Retrieve original tensors
recovered = ragged_array.to_numpy()

# All tensors maintain their original shape and dtype
```

## API

### `RaggedTensorArray`

A PyArrow `ExtensionArray` for storing ragged tensor data.

- **`from_numpy(tensors: list[np.ndarray])`**: Create a ragged array from a sequence of numpy arrays. All arrays must have matching trailing dimensions.
- **`to_numpy()`**: Convert the ragged array back to a list of numpy arrays.

### `RaggedTensorType`

The PyArrow `ExtensionType` that defines the schema for ragged tensors.

- **`inner_shape`**: Property storing the fixed trailing dimensions.
- **`numpy_dtype`**: Property storing the numpy dtype of the tensor data.
