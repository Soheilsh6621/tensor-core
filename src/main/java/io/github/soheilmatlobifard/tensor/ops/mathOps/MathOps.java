package io.github.soheilmatlobifard.tensor.ops.mathOps;

import io.github.soheilmatlobifard.tensor.Tensor;

public class MathOps {

    /**
     * Computes the absolute value of each element in the tensor (in-place).
     *
     * @param tensor the input tensor
     */
    public static void abs(Tensor tensor) {
        float[] data = tensor.getData();
        for (int i = 0; i < data.length; i++) {
            data[i] = Math.abs(data[i]);
        }
        tensor.setData(data);
    }

    /**
     * Computes the square root of each element in the tensor (in-place).
     *
     * @param tensor the input tensor
     * @throws IllegalArgumentException if any element is negative
     */
    public static void sqrt(Tensor tensor) {
        float[] data = tensor.getData();
        for (int i = 0; i < data.length; i++) {
            if (data[i] < 0) {
                throw new IllegalArgumentException(
                    "Cannot compute sqrt of negative value " + data[i] + " at index " + i
                );
            }
            data[i] = (float) Math.sqrt(data[i]);
        }
        tensor.setData(data);
    }

    /**
     * Raises each element of the tensor to the specified power (in-place).
     *
     * @param tensor the input tensor
     * @param exp    the exponent
     */
    public static void pow(Tensor tensor, float exp) {
        float[] data = tensor.getData();
        for (int i = 0; i < data.length; i++) {
            data[i] = (float) Math.pow(data[i], exp);
        }
        tensor.setData(data);
    }

    /**
     * Computes e raised to each element of the tensor (in-place).
     *
     * @param tensor the input tensor
     */
    public static void exp(Tensor tensor) {
        float[] data = tensor.getData();
        for (int i = 0; i < data.length; i++) {
            data[i] = (float) Math.exp(data[i]);
        }
        tensor.setData(data);
    }

    /**
     * Computes the natural logarithm of each element (in-place).
     *
     * @param tensor the input tensor
     * @throws IllegalArgumentException if any element is non-positive
     */
    public static void log(Tensor tensor) {
        float[] data = tensor.getData();
        for (int i = 0; i < data.length; i++) {
            if (data[i] <= 0) {
                throw new IllegalArgumentException(
                    "Cannot compute log of non-positive value " + data[i] + " at index " + i
                );
            }
            data[i] = (float) Math.log(data[i]);
        }
        tensor.setData(data);
    }

    /**
     * Computes the base-10 logarithm of each element (in-place).
     *
     * @param tensor the input tensor
     * @throws IllegalArgumentException if any element is non-positive
     */
    public static void log10(Tensor tensor) {
        float[] data = tensor.getData();
        for (int i = 0; i < data.length; i++) {
            if (data[i] <= 0) {
                throw new IllegalArgumentException(
                    "Cannot compute log10 of non-positive value " + data[i] + " at index " + i
                );
            }
            data[i] = (float) Math.log10(data[i]);
        }
        tensor.setData(data);
    }

    /**
     * Computes the sine of each element (in-place).
     *
     * @param tensor the input tensor
     */
    public static void sin(Tensor tensor) {
        float[] data = tensor.getData();
        for (int i = 0; i < data.length; i++) {
            data[i] = (float) Math.sin(data[i]);
        }
        tensor.setData(data);
    }

    /**
     * Computes the cosine of each element (in-place).
     *
     * @param tensor the input tensor
     */
    public static void cos(Tensor tensor) {
        float[] data = tensor.getData();
        for (int i = 0; i < data.length; i++) {
            data[i] = (float) Math.cos(data[i]);
        }
        tensor.setData(data);
    }

    /**
     * Computes the tangent of each element (in-place).
     *
     * @param tensor the input tensor
     */
    public static void tan(Tensor tensor) {
        float[] data = tensor.getData();
        for (int i = 0; i < data.length; i++) {
            data[i] = (float) Math.tan(data[i]);
        }
        tensor.setData(data);
    }

    /**
     * Rounds each element down to the nearest integer (in-place).
     *
     * @param tensor the input tensor
     */
    public static void floor(Tensor tensor) {
        float[] data = tensor.getData();
        for (int i = 0; i < data.length; i++) {
            data[i] = (float) Math.floor(data[i]);
        }
        tensor.setData(data);
    }

    /**
     * Rounds each element up to the nearest integer (in-place).
     *
     * @param tensor the input tensor
     */
    public static void ceil(Tensor tensor) {
        float[] data = tensor.getData();
        for (int i = 0; i < data.length; i++) {
            data[i] = (float) Math.ceil(data[i]);
        }
        tensor.setData(data);
    }

    /**
     * Rounds each element to the nearest integer (in-place).
     *
     * @param tensor the input tensor
     */
    public static void round(Tensor tensor) {
        float[] data = tensor.getData();
        for (int i = 0; i < data.length; i++) {
            data[i] = (float) Math.round(data[i]);
        }
        tensor.setData(data);
    }

    /**
     * Clamps each element to the range [min, max] (in-place).
     *
     * @param tensor the input tensor
     * @param min    the lower bound
     * @param max    the upper bound
     */
    public static void clip(Tensor tensor, float min, float max) {
        float[] data = tensor.getData();
        for (int i = 0; i < data.length; i++) {
            if (data[i] < min) data[i] = min;
            else if (data[i] > max) data[i] = max;
        }
        tensor.setData(data);
    }

    /**
     * Computes the signum of each element (in-place).
     * Returns -1 for negative, 0 for zero, 1 for positive.
     *
     * @param tensor the input tensor
     */
    public static void sign(Tensor tensor) {
        float[] data = tensor.getData();
        for (int i = 0; i < data.length; i++) {
            if (data[i] > 0) data[i] = 1;
            else if (data[i] < 0) data[i] = -1;
        }
        tensor.setData(data);
    }


}
