# Tensor-Core

Tensor-Core is a lightweight, open-source Java library for working with tensors and performing linear algebra, statistics, math, indexing, and other operations on them.

Inspired by libraries like NumPy (Python) and PyTorch, this project aims to provide a clean, intuitive API for tensor computations in the Java ecosystem.

---

## Project Goal

The primary goal of Tensor-Core is to offer a fluent, object-oriented API for creating, manipulating, and computing on tensors in Java. It is suitable for:

- Implementing machine learning algorithms from scratch
- Numerical and matrix computations
- Educational and research projects in data science
- Any other scenario where tensor operations are needed in Java

---

## Project Structure

Operations are divided into two main categories:

1. **Unary Operations** — operate on a single tensor.
2. **Binary Operations** — operate on two tensors.

---

### 1. Unary Operations

#### ShapeOps

Shape and dimension manipulation operations.

| Method | Description |
|---|---|
| `reshape(int... newShape)` | Reshape the tensor (supports -1 inference) |
| `flatten()` | Flatten to a 1D vector |
| `ravel()` | Flatten to a 1D vector (avoid copy when possible) |
| `transpose(int... newAxis)` | Permute tensor axes |
| `permute(int... dims)` | Permute axes (alias for transpose) |
| `squeeze()` | Remove all dimensions of size 1 |
| `unsqueeze(int dim)` | Insert a dimension of size 1 at a given position |
| `expandDims(int dim)` | Alias for unsqueeze |

#### StatisticsOps

Statistical operations over tensor elements.

| Method | Description |
|---|---|
| `sum()` | Sum of all elements |
| `mean()` | Mean of all elements |
| `min()` | Minimum value |
| `max()` | Maximum value |
| `argMin()` | Index of the minimum value |
| `argMax()` | Index of the maximum value |
| `variance()` | Variance |
| `std()` | Standard deviation |
| `prod()` | Product of all elements |
| `norm()` | L2 norm |

#### MathOps

Element-wise mathematical operations.

| Method | Description |
|---|---|
| `abs()` | Absolute value |
| `sqrt()` | Square root |
| `pow(float exp)` | Power |
| `exp()` | e raised to each element |
| `log()` | Natural logarithm |
| `log10()` | Base-10 logarithm |
| `sin()` | Sine |
| `cos()` | Cosine |
| `tan()` | Tangent |
| `floor()` | Round down |
| `ceil()` | Round up |
| `round()` | Round to nearest |
| `clip(float min, float max)` | Clamp values to a range |
| `sign()` | Signum (-1, 0, or 1) |

#### IndexingOps

Indexing and access operations.

| Method | Description |
|---|---|
| `slice(int... ranges)` | Slice a sub-tensor |
| `select(int dim, int index)` | Select a sub-tensor along one dimension |
| `take(int... indices)` | Gather elements by 1D indices |
| `get(int... indices)` | Get value at a specific index |
| `set(float value, int... indices)` | Set value at a specific index |
| `put(int[] indices, Tensor other)` | Place another tensor at a given position |
| `mask(Tensor condition)` | Filter elements by a boolean condition |

#### UtilityOps

Utility and helper operations.

| Method | Description |
|---|---|
| `copy()` | Shallow copy of the tensor |
| `clone()` | Deep copy of the tensor |
| `fill(float value)` | Fill all elements with a given value |
| `equals(Tensor other)` | Check equality of two tensors |
| `toArray()` | Convert to a multi-dimensional array |
| `toString()` | String representation of the tensor |

---

### 2. Binary Operations

These operations operate on two tensors.

#### ArithmeticOps

Element-wise arithmetic operations.

| Method | Description |
|---|---|
| `add(Tensor a, Tensor b)` | Addition |
| `subtract(Tensor a, Tensor b)` | Subtraction |
| `multiply(Tensor a, Tensor b)` | Element-wise multiplication |
| `divide(Tensor a, Tensor b)` | Division |
| `mod(Tensor a, Tensor b)` | Modulo |
| `maximum(Tensor a, Tensor b)` | Element-wise maximum |
| `minimum(Tensor a, Tensor b)` | Element-wise minimum |

#### LinearAlgebraOps

Linear algebra operations.

| Method | Description |
|---|---|
| `dot(Tensor a, Tensor b)` | Dot product (vectors) |
| `matmul(Tensor a, Tensor b)` | Matrix multiplication |
| `outer(Tensor a, Tensor b)` | Outer product (vectors) |
| `cross(Tensor a, Tensor b)` | Cross product (3D vectors) |
| `tensorProduct(Tensor a, Tensor b)` | Tensor product |
| `kronecker(Tensor a, Tensor b)` | Kronecker product |
| `contract(Tensor a, int... dims)` | Tensor contraction |

#### ComparisonOps

Element-wise comparison operations.

| Method | Description |
|---|---|
| `equal(Tensor a, Tensor b)` | Equality |
| `notEqual(Tensor a, Tensor b)` | Inequality |
| `greater(Tensor a, Tensor b)` | Greater than |
| `less(Tensor a, Tensor b)` | Less than |
| `greaterEqual(Tensor a, Tensor b)` | Greater than or equal |
| `lessEqual(Tensor a, Tensor b)` | Less than or equal |

#### MergeOps

Operations for joining and splitting tensors.

| Method | Description |
|---|---|
| `concat(Tensor... tensors)` | Concatenate tensors along a given axis |
| `stack(Tensor... tensors)` | Stack tensors along a new axis |
| `vstack(Tensor... tensors)` | Stack vertically (rows) |
| `hstack(Tensor... tensors)` | Stack horizontally (columns) |
| `split(Tensor tensor, int... splits)` | Split a tensor into pieces |

#### BroadcastOps

Broadcast operations.

| Method | Description |
|---|---|
| `broadcastTo(Tensor tensor, int... shape)` | Broadcast a tensor to a target shape |
| `broadcastAdd(Tensor a, Tensor b)` | Addition with automatic broadcasting |
| `broadcastMultiply(Tensor a, Tensor b)` | Multiplication with automatic broadcasting |

---

## Quick Example

```java
Tensor a = Tensor.matrix(new float[][]{{1, 2}, {3, 4}});
Tensor b = Tensor.vector(1, 2);

// reshape
Tensor r = a.reshape(4);            // [1, 2, 3, 4]

// statistics
float m = a.mean();                  // 2.5
float v = a.variance();              // 1.25

// transpose
Tensor t = a.transpose(1, 0);        // [[1, 3], [2, 4]]

// matrix multiplication
Tensor result = LinearAlgebraOps.matmul(a, b);
```

---

## License

This project is licensed under the MIT License.
