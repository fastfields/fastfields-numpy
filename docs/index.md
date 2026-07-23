# fastfields-numpy

`fastfields-numpy` is a thin, user-friendly **numpy** interface over the `fastfields.dlpack` bindings. The bindings themselves operate in place / write through pre-allocated outputs; these wrappers take numpy arrays and **return** freshly allocated numpy arrays (the input is never clobbered unless you pass `inplace=True`).

## Installation

```bash
pip install fastfields-numpy
```

## Usage

```python
import numpy as np
import fastfields.numpy as ff

x = np.array([0, np.inf, np.inf, 0, np.inf], dtype=np.float32)
d = ff.euclidean_distance_transform(x)      # squared EDT along the last axis
```

See the [API reference](api/index.md) for the full list of operations.
