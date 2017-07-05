package io.github.marcelovca90.ml;

import java.util.Arrays;

import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.back.Backpropagation;

import io.github.marcelovca90.common.MessageLabel;
import io.github.marcelovca90.helper.DataSetHelper;
import io.github.marcelovca90.helper.MethodHelper;
import io.github.marcelovca90.math.ActivationLogSig;
import io.github.marcelovca90.math.ActivationTanSig;

public class MLP_BPROP extends AbstractClassifier
{
    @Override
    public void train(BasicMLDataSet trainingSet, BasicMLDataSet validationSet)
    {
        int inputCount = trainingSet.get(0).getInput().size();
        int hiddenCount = MethodHelper.calculateHiddenLayerSize(inputCount, trainingSet.size());
        int outputCount = trainingSet.get(0).getIdeal().size();

        BasicNetwork network = new BasicNetwork();
        network.addLayer(new BasicLayer(new ActivationTanSig(), true, inputCount));
        network.addLayer(new BasicLayer(new ActivationTanSig(), true, hiddenCount));
        network.addLayer(new BasicLayer(new ActivationTanSig(), true, hiddenCount));
        network.addLayer(new BasicLayer(new ActivationLogSig(), false, outputCount));
        network.getStructure().finalizeStructure();
        network.reset(seed);

        Backpropagation backpropagation = new Backpropagation(network, trainingSet, 1e-3, 0.9);
        backpropagation.setBatchSize(0);
        backpropagation.setThreadCount(0);

        double trainingErrorBefore = Double.MAX_VALUE, trainingErrorAfter = Double.MAX_VALUE;
        double validationErrorBefore = Double.MAX_VALUE, validationErrorAfter = Double.MAX_VALUE;

        do
        {
            trainingErrorBefore = backpropagation.getError();
            validationErrorBefore = validationErrorAfter;

            backpropagation.iteration(20);

            trainingErrorAfter = backpropagation.getError();
            validationErrorAfter = network.calculateError(validationSet);

            if (trainingErrorAfter < trainingErrorBefore)
            {
                backpropagation.setLearningRate(1.02 * backpropagation.getLearningRate());
                backpropagation.setMomentum(backpropagation.getMomentum());
            }
            else
            {
                backpropagation.setLearningRate(0.50 * backpropagation.getLearningRate());
                backpropagation.setMomentum(0);
            }

            /*
             * logger.debug(String.format("Iteration #%d\tLR = %3.3e\tMO = %.6f\tvError = %.12f", backpropagation.getIteration(),
             * backpropagation.getLearningRate(), backpropagation.getMomentum(), validationErrorAfter));
             */

        } while (validationErrorAfter < validationErrorBefore);

        backpropagation.finishTraining();

        this.method = network;
    }

    @Override
    public void test(BasicMLDataSet testSet)
    {
        int hamCount = 0, hamCorrect = 0;
        int spamCount = 0, spamCorrect = 0;
        BasicNetwork network = ((BasicNetwork) this.method);

        for (MLDataPair pair : testSet)
        {
            MLData input = pair.getInput();
            MLData ideal = pair.getIdeal();
            MLData output = network.compute(input);

            if (MethodHelper.infer(ideal.getData()) == MessageLabel.HAM)
            {
                hamCount++;
                if (MethodHelper.infer(output.getData()) == MessageLabel.HAM)
                {
                    hamCorrect++;
                }
            }
            else if (MethodHelper.infer(ideal.getData()) == MessageLabel.SPAM)
            {
                spamCount++;
                if (MethodHelper.infer(output.getData()) == MessageLabel.SPAM)
                {
                    spamCorrect++;
                }
            }
        }

        double hamPrecision = 100.0 * hamCorrect / hamCount;
        double spamPrecision = 100.0 * spamCorrect / spamCount;
        double precision = 100.0 * (hamCorrect + spamCorrect) / (hamCount + spamCount);
        String structure = Arrays.toString(network.getFlat().getLayerCounts());

        logger.info(String.format("%s @ %d\t%s\tHP: %.2f%% (%d/%d)\tSP: %.2f%% (%d/%d)\tGP: %.2f%%", structure, seed,
                folder.replace(DataSetHelper.BASE_FOLDER, ""), hamPrecision, hamCorrect, hamCount, spamPrecision, spamCorrect, spamCount, precision));
    }
}
