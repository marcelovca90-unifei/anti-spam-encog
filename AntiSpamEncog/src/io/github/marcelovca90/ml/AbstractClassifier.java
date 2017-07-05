package io.github.marcelovca90.ml;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.encog.ml.data.basic.BasicMLDataSet;

public abstract class AbstractClassifier
{
    protected Logger logger;

    protected String folder;

    protected int seed;

    protected Object method;

    public AbstractClassifier()
    {
        this.logger = LogManager.getLogger(this.getClass());
    }

    public void initialize(String folder, int seed)
    {
        this.folder = folder;
        this.seed = seed;
        this.method = null;
    }

    public abstract void train(BasicMLDataSet trainingSet, BasicMLDataSet validationSet);

    public abstract void test(BasicMLDataSet testSet);
}
