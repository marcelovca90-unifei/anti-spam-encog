package io.github.marcelovca90.main;

import java.io.File;

import org.encog.Encog;
import org.encog.ml.data.basic.BasicMLDataSet;

import io.github.marcelovca90.common.MessageDataSet;
import io.github.marcelovca90.helper.DataSetHelper;
import io.github.marcelovca90.helper.PrimeHelper;
import io.github.marcelovca90.ml.AbstractClassifier;
import io.github.marcelovca90.ml.MLP_BPROP;
import io.github.marcelovca90.ml.MLP_RPROP;
import io.github.marcelovca90.ml.NEAT;
import io.github.marcelovca90.ml.RBF_QPROP;
import io.github.marcelovca90.ml.SVM_RBF;

public class Main
{
    public static void main(String[] args) throws Exception
    {
        Class<?>[] classes = new Class[] { MLP_BPROP.class, MLP_RPROP.class, NEAT.class, RBF_QPROP.class, SVM_RBF.class };

        for (Class<?> clazz : classes)
        {
            for (String folder : DataSetHelper.getFolders())
            {
                File hamFile = new File(folder + "/ham");
                File spamFile = new File(folder + "/spam");
                int seed = PrimeHelper.getMiddlePrime();

                MessageDataSet dataSet = new MessageDataSet(hamFile, spamFile);
                MessageDataSet dataSubset = null;

                dataSet.replicate(seed);
                dataSet.shuffle(seed);

                dataSubset = dataSet.getSubset(0, 40);
                BasicMLDataSet trainingSet = new BasicMLDataSet(dataSubset.getInputDataAsPrimitiveMatrix(), dataSubset.getOutputDataAsPrimitiveMatrix());

                dataSubset = dataSet.getSubset(40, 60);
                BasicMLDataSet validationSet = new BasicMLDataSet(dataSubset.getInputDataAsPrimitiveMatrix(), dataSubset.getOutputDataAsPrimitiveMatrix());

                dataSubset = dataSet.getSubset(60, 100);
                BasicMLDataSet testSet = new BasicMLDataSet(dataSubset.getInputDataAsPrimitiveMatrix(), dataSubset.getOutputDataAsPrimitiveMatrix());

                AbstractClassifier classifier = (AbstractClassifier) clazz.newInstance();

                classifier.initialize(folder, seed);
                classifier.train(trainingSet, validationSet);
                classifier.test(testSet);
            }
        }

        Encog.getInstance().shutdown();
    }
}
