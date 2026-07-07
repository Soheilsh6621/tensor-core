package io.github.soheilmatlobifard.tensor;

public class App {

    public static void main(String[] args) {
        Tensor vector = Tensor.vector(1, 2, 3);
        System.out.println("Vector: " + vector);

        Tensor matrix = Tensor.matrix(new float[][]{
            {1, 2},
            {3, 4},
            {5, 6}
        });
        System.out.println("Matrix:\n" + matrix);

        Tensor zeros = Tensor.zeros(2, 3);
        System.out.println("Zeros:\n" + zeros);

        Tensor ones = Tensor.ones(2, 2);
        System.out.println("Ones:\n" + ones);

        Tensor identity = Tensor.eye(3);
        System.out.println("Identity:\n" + identity);

        Tensor a = Tensor.matrix(new float[][]{{1, 2}, {3, 4}});
        Tensor b = Tensor.matrix(new float[][]{{5, 6}, {7, 8}});

        Tensor t3d = new Tensor(2, 3, 4);
    }
}
