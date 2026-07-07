package io.github.soheilmatlobifard.tensor.ops.statisticsOps;

import io.github.soheilmatlobifard.tensor.Tensor;

public class StatisticsOps {

    /**
     * Computes the sum of all elements in the tensor.
     *
     * @param tensor the input tensor
     * @return the sum of all elements
     */
    public static float sum(Tensor tensor) {
        float s = 0;

        for (int i = 0; i < tensor.getSize(); i++) {
            s += tensor.get(i);
        }

        return s;
    }


    /**
     * Computes the sum of elements along axis 0 (rows) for 2D matrices.
     * Returns a 1D tensor of length equal to the number of columns,
     * where each element is the sum of its column across all rows.
     *
     * @param tensor the input matrix (rank 2)
     * @return a 1D tensor containing column-wise sums
     * @throws IllegalArgumentException if the tensor rank is not 2
     */
    public static Tensor sumAxis0(Tensor tensor) {
        if (tensor.getRank() != 2) {
            throw new IllegalArgumentException(
                    "sumAxis0 only supports rank 2 tensors (matrices)"
            );
        }

        int rows = tensor.getShape()[0];
        int cols = tensor.getShape()[1];

        float[] result = new float[cols];

        for (int i = 0; i < rows; i++) {

            for (int j = 0; j < cols; j++) {

                result[j] += tensor.getData()[i * cols + j];
            }

        }

        return new Tensor(result, cols);
    }

    /**
     * Computes the mean (average) of all elements in the tensor.
     *
     * @return the average value of all elements
     */
    public static float mean(Tensor tensor) {
        float sum = 0;

        for (int i = 0; i < tensor.getSize(); i++) {
            sum += tensor.getShape()[i];
        }

        return sum / tensor.getSize();
    }

    /**
     * Computes the mean of elements along axis 0 (rows) for 2D matrices.
     * Returns a 1D tensor of length equal to the number of columns,
     * where each element is the average of its column across all rows.
     *
     * @param tensor the input matrix (rank 2)
     * @return a 1D tensor containing column-wise means
     * @throws IllegalArgumentException if the tensor rank is not 2
     */
    public static Tensor meanAxis0(Tensor tensor) {
        if (tensor.getRank() != 2) {
            throw new IllegalArgumentException(
                    "meanAxis0 only supports rank 2 tensors (matrices)"
            );
        }

        int rows = tensor.getShape()[0];
        int cols = tensor.getShape()[1];

        float[] result = new float[cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[j] += tensor.getData()[i * cols + j];
            }
        }

        for (int j = 0; j < cols; j++) {
            result[j] /= rows;
        }

        return new Tensor(result, cols);
    }


    /**
     * Computes the minimum value of all elements in the tensor.
     * Iterates through all elements and returns the smallest value found.
     *
     * @param tensor the input tensor whose elements are scanned
     * @return the minimum value in the tensor
     * @throws IllegalStateException if the tensor is empty (size == 0)
     */
    public static float min(Tensor tensor) {
        if (tensor.getSize() == 0) {
            throw new IllegalStateException("Cannot compute min of empty tensor");
        }

        float m = tensor.getData()[0];

        for (int i = 1; i < tensor.getSize() ; i++) {

            if (tensor.getData()[i] < m) {

                m = tensor.getData()[i];
            }
        }

        return m;
    }


    /**
     * Computes the maximum value of all elements in the tensor.
     * Iterates through all elements and returns the largest value found.
     *
     * @param tensor the input tensor whose elements are scanned
     * @return the maximum value in the tensor
     * @throws IllegalStateException if the tensor is empty (size == 0)
     */
    public static float max(Tensor tensor) {
        if (tensor.getSize() == 0) {
            throw new IllegalStateException("Cannot compute max of empty tensor");
        }

        float m = tensor.getData()[0];

        for (int i = 1; i < tensor.getSize(); i++) {
            if (tensor.getData()[i] > m) {
                m = tensor.getData()[i];
            }
        }

        return m;
    }

    /**
     * Returns the index of the maximum value in the tensor.
     * Iterates through all elements and tracks the position of the largest value.
     * If multiple elements have the same maximum value, the first occurrence is returned.
     *
     * @param tensor the input tensor to search
     * @return the index of the maximum value
     * @throws IllegalStateException if the tensor is empty (size == 0)
     */
    public static int argMax(Tensor tensor) {
        if (tensor.getSize() == 0) {
            throw new IllegalStateException("Cannot compute argMax of empty tensor");
        }

        float[] data = tensor.getData();

        int maxIndex = 0;
        float maxValue = data[0];

        for (int i = 1; i < tensor.getSize(); i++) {
            if (data[i] > maxValue) {
                maxValue = data[i];
                maxIndex = i;
            }
        }

        return maxIndex;
    }


    /**
     * Returns the index of the minimum value in the tensor.
     * Iterates through all elements and tracks the position of the smallest value.
     * If multiple elements have the same minimum value, the first occurrence is returned.
     *
     * @param tensor the input tensor to search
     * @return the index of the minimum value
     * @throws IllegalStateException if the tensor is empty (size == 0)
     */
    public static int argMin(Tensor tensor) {
        if (tensor.getSize() == 0) {
            throw new IllegalStateException("Cannot compute argMin of empty tensor");
        }

        float[] data = tensor.getData();

        int minIndex = 0;
        float minValue = data[0];

        for (int i = 1; i < tensor.getSize(); i++) {
            if (data[i] < minValue) {
                minValue = data[i];
                minIndex = i;
            }
        }

        return minIndex;
    }


    /**
     * Computes the standard deviation of all elements in the tensor.
     * Standard deviation is the square root of the variance,
     * measuring the spread of values around the mean.
     *
     * Formula: std = sqrt(variance)
     *
     * @param tensor input tensor (must not be empty)
     * @return standard deviation of tensor elements
     * @throws IllegalStateException if tensor is empty
     */
    public static float std(Tensor tensor) {
        return (float) Math.sqrt(variance(tensor));
    }

    /**
     * Computes the product of all elements in the tensor.
     *
     * @param tensor the input tensor
     * @return the product of all elements
     */
    public static float prod(Tensor tensor) {
        float p = 1;

        for (int i = 0; i < tensor.getSize(); i++) {
            p *= tensor.get(i);
        }

        return p;
    }

    /**
     * Computes the L2 norm (Euclidean norm) of the tensor.
     * The L2 norm is the square root of the sum of squared elements.
     *
     * Formula: ||tensor||₂ = sqrt(Σ x_i²)
     *
     * @param tensor the input tensor
     * @return the L2 norm of the tensor
     */
    public static float norm(Tensor tensor) {
        float sumSquares = 0;

        for (int i = 0; i < tensor.getSize(); i++) {
            float v = tensor.get(i);
            sumSquares += v * v;
        }

        return (float) Math.sqrt(sumSquares);
    }

    /**
     * Computes the variance of all elements in the tensor.
     * Variance measures how far the values are spread from the mean.
     *
     * Formula:
     * variance = (1 / n) * Σ (x_i - mean)^2
     *
     * @param tensor input tensor (must not be empty)
     * @return variance value of tensor elements
     * @throws IllegalStateException if tensor is empty
     */
    public static float variance(Tensor tensor) {

        float[] data = tensor.getData();

        if (data.length == 0) {
            throw new IllegalStateException("Cannot compute variance of empty tensor");
        }

        float mean = StatisticsOps.mean(tensor);

        float sum = 0;

        for (int i = 0; i < data.length; i++) {
            float diff = data[i] - mean;
            sum += diff * diff;
        }

        return sum / data.length;
    }

}
