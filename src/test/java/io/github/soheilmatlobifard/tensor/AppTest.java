package io.github.soheilmatlobifard.tensor;

import junit.framework.Test;
import junit.framework.TestCase;
import junit.framework.TestSuite;

public class AppTest extends TestCase {

    public AppTest(String testName) {
        super(testName);
    }

    public static Test suite() {
        return new TestSuite(AppTest.class);
    }

    public void testZeros() {
        Tensor z = Tensor.zeros(2, 3);
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 3; j++)
                assertEquals(0.0f, z.get(i, j));
    }

    public void testOnes() {
        Tensor o = Tensor.ones(2, 2);
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                assertEquals(1.0f, o.get(i, j));
    }

    public void testEye() {
        Tensor I = Tensor.eye(3);
        assertEquals(1.0f, I.get(0, 0));
        assertEquals(1.0f, I.get(1, 1));
        assertEquals(1.0f, I.get(2, 2));
        assertEquals(0.0f, I.get(0, 1));
        assertEquals(0.0f, I.get(1, 0));
    }



    public void testSum() {
        Tensor v = Tensor.vector(1, 2, 3, 4);
        assertEquals(10.0f, v.sum());
    }

    public void testSumAxis0() {
        Tensor m = Tensor.matrix(new float[][]{{1, 2}, {3, 4}});
        Tensor r = m.sumAxis0();
        assertEquals(4.0f, r.get(0));
        assertEquals(6.0f, r.get(1));
    }

    public void testMean() {
        Tensor v = Tensor.vector(2, 4, 6);
        assertEquals(4.0f, v.mean());
    }

    public void testMeanAxis0() {
        Tensor m = Tensor.matrix(new float[][]{{1, 2}, {3, 4}});
        Tensor r = m.meanAxis0();
        assertEquals(2.0f, r.get(0));
        assertEquals(3.0f, r.get(1));
    }

    public void testProd() {
        Tensor v = Tensor.vector(2, 3, 4);
        assertEquals(24.0f, v.prod());
    }

    public void testNorm() {
        Tensor v = Tensor.vector(3, 4);
        assertEquals(5.0f, v.norm());
    }

    public void testStd() {
        Tensor v = Tensor.vector(1, 2, 3);
        float expected = (float) Math.sqrt((1 + 0 + 1) / 3.0);
        assertEquals(expected, v.std(), 1e-6f);
    }

    public void testVariance() {
        Tensor v = Tensor.vector(1, 2, 3);
        float expected = (float) ((1 + 0 + 1) / 3.0);
        assertEquals(expected, v.variance(), 1e-6f);
    }

    public void testMin() {
        Tensor v = Tensor.vector(5, 2, 8, 1);
        assertEquals(1.0f, v.min());
    }

    public void testMax() {
        Tensor v = Tensor.vector(5, 2, 8, 1);
        assertEquals(8.0f, v.max());
    }

    public void testArgMax() {
        Tensor v = Tensor.vector(5, 2, 8, 1);
        assertEquals(2, v.argMax());
    }

    public void testArgMin() {
        Tensor v = Tensor.vector(5, 2, 8, 1);
        assertEquals(3, v.argMin());
    }
}
