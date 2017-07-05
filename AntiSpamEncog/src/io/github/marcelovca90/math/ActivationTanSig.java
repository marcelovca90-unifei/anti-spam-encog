package io.github.marcelovca90.math;

import org.encog.engine.network.activation.ActivationFunction;

public class ActivationTanSig implements ActivationFunction
{
    private static final long serialVersionUID = 1L;

    @Override
    public void activationFunction(double[] d, int start, int size)
    {
        for (int i = 0; i < d.length; i++)
            d[i] = 2.0 / (1.0 + Math.exp(-2.0 * d[i])) - 1.0;
    }

    @Override
    public double derivativeFunction(double b, double a)
    {
        return (4.0 * Math.exp(2.0 * b)) / (Math.pow(Math.exp(2.0 * b) + 1.0, 2.0));
    }

    @Override
    public boolean hasDerivative()
    {
        return true;
    }

    @Override
    public double[] getParams()
    {
        return new double[0];
    }

    @Override
    public void setParam(int index, double value)
    {
        return;
    }

    @Override
    public String[] getParamNames()
    {
        return new String[0];
    }

    @Override
    public String getFactoryCode()
    {
        return this.getClass().getName();
    }

    @Override
    public ActivationFunction clone()
    {
        return new ActivationTanSig();
    }

    public String getLabel()
    {
        return this.getClass().getName();
    }
}
