package xyz.marcelo.ml;

import java.util.Arrays;

import org.encog.mathutil.rbf.RBFEnum;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.networks.training.propagation.quick.QuickPropagation;
import org.encog.neural.rbf.RBFNetwork;

import xyz.marcelo.common.MessageLabel;
import xyz.marcelo.helper.DataSetHelper;
import xyz.marcelo.helper.MethodHelper;

public class RBF_QPROP extends AbstractClassifier
{
    @Override
    public void train(BasicMLDataSet trainingSet, BasicMLDataSet validationSet)
    {
        int inputCount = trainingSet.get(0).getInput().size();
        int hiddenCount = MethodHelper.calculateHiddenLayerSize(inputCount, trainingSet.size());
        int outputCount = trainingSet.get(0).getIdeal().size();

        RBFNetwork network = new RBFNetwork(inputCount, hiddenCount, outputCount, RBFEnum.Gaussian);
        network.reset();

        QuickPropagation quickPropagation = new QuickPropagation(network, trainingSet);
        quickPropagation.setBatchSize(0);
        quickPropagation.setThreadCount(0);

        double validationErrorBefore = Double.MAX_VALUE, validationErrorAfter = Double.MAX_VALUE;

        do
        {
            validationErrorBefore = validationErrorAfter;

            quickPropagation.iteration(20);

            validationErrorAfter = network.calculateError(validationSet);

            // logger.debug(String.format("Iteration #%d\tvError = %.12f", quickPropagation.getIteration(), validationErrorAfter));

        } while (validationErrorAfter < validationErrorBefore);

        quickPropagation.finishTraining();

        this.method = network;
    }

    @Override
    public void test(BasicMLDataSet testSet)
    {
        int hamCount = 0, hamCorrect = 0;
        int spamCount = 0, spamCorrect = 0;
        RBFNetwork network = ((RBFNetwork) this.method);

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
