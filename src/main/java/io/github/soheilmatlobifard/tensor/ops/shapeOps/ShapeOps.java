package io.github.soheilmatlobifard.tensor.ops.shapeOps;

import io.github.soheilmatlobifard.tensor.Tensor;

public class ShapeOps {

    /**
     * Reshapes a tensor to a new shape.
     * One dimension may be inferred by passing -1.
     *
     * @param tensor   the input tensor
     * @param newShape the target shape (at most one -1 for automatic inference)
     * @return a new tensor with the given shape sharing the same data
     * @throws IllegalArgumentException if more than one dimension is -1,
     *                                  or if the total number of elements does not match
     */
    public static Tensor reshape(Tensor tensor, int... newShape) {
        int inferred = -1;
        int product = 1;
        for (int i = 0; i < newShape.length; i++) {
            if (newShape[i] == -1) {
                if (inferred != -1) {
                    throw new IllegalArgumentException("Only one dimension can be inferred (-1)");
                }
                inferred = i;
            } else {
                product *= newShape[i];
            }
        }
        if (inferred != -1) {
            int size = tensor.getSize();
            if (size % product != 0) {
                throw new IllegalArgumentException(
                    "Tensor size " + size + " is not divisible by product of known dimensions " + product
                );
            }
            newShape[inferred] = size / product;
        }
        return new Tensor(tensor.getData(), newShape);
    }

    private static int flatIndex(int[] indices, int[] shape) {
        int idx = 0;
        for (int i = 0; i < shape.length; i++) {
            idx = idx * shape[i] + indices[i];
        }
        return idx;
    }

    private static int[] unravelIndex(int flatIndex, int[] shape) {
        int[] indices = new int[shape.length];
        for (int i = shape.length - 1; i >= 0; i--) {
            indices[i] = flatIndex % shape[i];
            flatIndex /= shape[i];
        }
        return indices;
    }

    /**
     * Flattens a tensor into a 1D tensor.
     *
     * @param tensor the input tensor
     * @return a 1D tensor containing all elements in row-major order
     */
    public static Tensor flatten(Tensor tensor) {
        return reshape(tensor, tensor.getSize());
    }


    /**
     * Permutes the axes of a tensor according to the specified axis order.
     *
     * @param tensor  the input tensor
     * @param newAxis the desired order of axes (must be a permutation of 0..rank-1)
     * @return a new tensor with the same data but permuted axes
     * @throws IllegalArgumentException if the number of axes does not match the tensor rank,
     *                                  if any axis is out of bounds, or if there are duplicate axes
     */
    public static Tensor transpose(Tensor tensor, int... newAxis) {

        int rank = tensor.getRank();


        if (newAxis.length != rank) {
            throw new IllegalArgumentException(
                    "Number of axes must match tensor rank. Expected "
                            + rank + " axes but got " + newAxis.length + "."
            );
        }


        boolean[] visited = new boolean[rank];

        for (int axis : newAxis) {

            if (axis < 0 || axis >= rank) {
                throw new IllegalArgumentException(
                        "Invalid axis " + axis +
                                ". Axis must be between 0 and " + (rank - 1)
                );
            }

            if (visited[axis]) {
                throw new IllegalArgumentException(
                        "Duplicate axis " + axis + " found in transpose."
                );
            }

            visited[axis] = true;
        }

        int[] oldShape = tensor.getShape();

        int[] newShape = new int[rank];

        for (int i = 0; i < rank; i++) {
            newShape[i] = oldShape[newAxis[i]];
        }

        float[] oldData = tensor.getData();

        float[] newData = new float[tensor.getSize()];

        for (int oldFlat = 0; oldFlat < oldData.length; oldFlat++) {

            int[] oldIndex = unravelIndex(oldFlat, oldShape);

            int[] newIndex = new int[rank];


            for (int i = 0; i < rank; i++) {

                newIndex[i] = oldIndex[newAxis[i]];

            }

            int newFlat = flatIndex(newIndex, newShape);

            newData[newFlat] = oldData[oldFlat];
        }

        return new Tensor(newData, newShape);
    }


}
