package io.github.soheilmatlobifard.tensor;

import java.util.Arrays;
import io.github.soheilmatlobifard.tensor.ops.statisticsOps.StatisticsOps;
import io.github.soheilmatlobifard.tensor.ops.shapeOps.ShapeOps;

public class Tensor {

    private final int[] shape;
    private final int rank;
    private final int size;
    private final float[] data;

    public Tensor(int... shape) {
        this.shape = shape.clone();
        this.rank = shape.length;
        this.size = computeSize(shape);
        this.data = new float[size];
    }

    public Tensor(float[] data, int... shape) {
        if (data.length != computeSize(shape)) {
            throw new IllegalArgumentException(
                "Data length " + data.length + " does not match shape " + Arrays.toString(shape)
            );
        }
        this.shape = shape.clone();
        this.rank = shape.length;
        this.size = data.length;
        this.data = data.clone();
    }

    public static Tensor zeros(int... shape) {
        return new Tensor(shape);
    }

    public static Tensor ones(int... shape) {
        Tensor t = new Tensor(shape);
        Arrays.fill(t.data, 1.0f);
        return t;
    }

    public static Tensor eye(int n) {
        Tensor t = new Tensor(n, n);
        for (int i = 0; i < n; i++) {
            t.set(1.0f, i, i);
        }
        return t;
    }

    public static Tensor vector(float... values) {
        return new Tensor(values, values.length);
    }

    public static Tensor matrix(float[][] values) {
        int rows = values.length;
        int cols = values[0].length;
        Tensor t = new Tensor(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                t.set(values[i][j], i, j);
            }
        }
        return t;
    }

    public int getRank() {
        return rank;
    }

    public int[] getShape() {
        return shape.clone();
    }

    public int getSize() {
        return size;
    }

    public float[] getData() {
        return data.clone();
    }

    public void setData(float[] data) {
        if (data.length != size) {
            throw new IllegalArgumentException(
                "Data length " + data.length + " does not match size " + size
            );
        }
        System.arraycopy(data, 0, this.data, 0, size);
    }


    int flatIndex(int... indices) {
        if (indices.length != rank) {
            throw new IllegalArgumentException(
                "Expected " + rank + " indices, got " + indices.length
            );
        }
        int idx = 0;
        for (int i = 0; i < rank; i++) {
            if (indices[i] < 0 || indices[i] >= shape[i]) {
                throw new IndexOutOfBoundsException(
                    "Index " + i + " = " + indices[i] + " out of bounds for shape " + Arrays.toString(shape)
                );
            }
            idx = idx * shape[i] + indices[i];
        }
        return idx;
    }

    public float get(int... indices) {
        return data[flatIndex(indices)];
    }

    public void set(float value, int... indices) {
        data[flatIndex(indices)] = value;
    }

    @Override
    public String toString() {
        if (rank == 0) {
            return Float.toString(data[0]);
        }
        if (rank == 1) {
            StringBuilder sb = new StringBuilder("[");
            for (int i = 0; i < shape[0]; i++) {
                if (i > 0) sb.append(", ");
                sb.append(data[i]);
            }
            sb.append("]");
            return sb.toString();
        }
        if (rank == 2) {
            StringBuilder sb = new StringBuilder("[");
            for (int i = 0; i < shape[0]; i++) {
                if (i > 0) sb.append(",\n ");
                sb.append("[");
                for (int j = 0; j < shape[1]; j++) {
                    if (j > 0) sb.append(", ");
                    sb.append(get(i, j));
                }
                sb.append("]");
            }
            sb.append("]");
            return sb.toString();
        }
        return "Tensor(shape=" + Arrays.toString(shape) + ")";
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Tensor)) return false;
        Tensor tensor = (Tensor) o;
        return Arrays.equals(shape, tensor.shape) && Arrays.equals(data, tensor.data);
    }

    @Override
    public int hashCode() {
        int result = Arrays.hashCode(shape);
        result = 31 * result + Arrays.hashCode(data);
        return result;
    }

    private static int computeSize(int[] shape) {
        int s = 1;
        for (int d : shape) {
            s *= d;
        }
        return s;
    }

    public float sum() {
        return StatisticsOps.sum(this);
    }

    public Tensor sumAxis0() {
        return StatisticsOps.sumAxis0(this);
    }

    public float std() {
        return StatisticsOps.std(this);
    }

    public float prod() {
        return StatisticsOps.prod(this);
    }

    public float norm() {
        return StatisticsOps.norm(this);
    }

    public float mean() {
        return StatisticsOps.mean(this);
    }

    public Tensor meanAxis0() {
        return StatisticsOps.meanAxis0(this);
    }

    public float min() {
        return StatisticsOps.min(this);
    }

    public float max() {
        return StatisticsOps.max(this);
    }

    public int argMax() {
        return StatisticsOps.argMax(this);
    }

    public int argMin() {
        return StatisticsOps.argMin(this);
    }

    public float variance() {
        return StatisticsOps.variance(this);
    }

    public Tensor reshape(int... newShape) {
        return ShapeOps.reshape(this, newShape);
    }

    public Tensor flatten() {
        return ShapeOps.flatten(this);
    }

    public Tensor transpose(int... newAxis) {
        return ShapeOps.transpose(this, newAxis);
    }
}
