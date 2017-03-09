package xyz.marcelo.ml;

import java.util.Arrays;

import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;

import xyz.marcelo.common.MessageLabel;
import xyz.marcelo.helper.DataSetHelper;
import xyz.marcelo.helper.MethodHelper;
import xyz.marcelo.math.ActivationLogSig;
import xyz.marcelo.math.ActivationTanSig;

public class MLP_RPROP extends AbstractClassifier
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

        ResilientPropagation resilientPropagation = new ResilientPropagation(network, trainingSet);
        resilientPropagation.setBatchSize(0);
        resilientPropagation.setThreadCount(0);

        double validationErrorBefore = Double.MAX_VALUE, validationErrorAfter = Double.MAX_VALUE;

        do
        {
            validationErrorBefore = validationErrorAfter;

            resilientPropagation.iteration(20);

            validationErrorAfter = network.calculateError(validationSet);

            // logger.debug(String.format("Iteration #%d\tvError = %.12f", resilientPropagation.getIteration(), validationErrorAfter));

        } while (validationErrorAfter < validationErrorBefore);

        resilientPropagation.finishTraining();

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
