package xyz.marcelo.ml;

import org.encog.ml.CalculateScore;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.ml.ea.train.basic.TrainEA;
import org.encog.neural.neat.NEATNetwork;
import org.encog.neural.neat.NEATPopulation;
import org.encog.neural.neat.NEATUtil;
import org.encog.neural.networks.training.TrainingSetScore;

import xyz.marcelo.common.MessageLabel;
import xyz.marcelo.helper.DataSetHelper;
import xyz.marcelo.helper.MethodHelper;

public class NEAT extends AbstractClassifier
{
    @Override
    public void train(BasicMLDataSet trainingSet, BasicMLDataSet validationSet)
    {
        int inputCount = trainingSet.get(0).getInput().size();
        int hiddenCount = MethodHelper.calculateHiddenLayerSize(inputCount, trainingSet.size());
        int outputCount = trainingSet.get(0).getIdeal().size();

        NEATNetwork network = null;

        NEATPopulation population = new NEATPopulation(inputCount, outputCount, hiddenCount);
        population.reset();

        CalculateScore score = new TrainingSetScore(trainingSet);

        TrainEA neatTrainer = NEATUtil.constructNEATTrainer(population, score);
        neatTrainer.setThreadCount(Runtime.getRuntime().availableProcessors());

        double validationErrorBefore = Double.MAX_VALUE, validationErrorAfter = Double.MAX_VALUE;

        do
        {
            validationErrorBefore = validationErrorAfter;

            neatTrainer.iteration(20);

            network = (NEATNetwork) neatTrainer.getCODEC().decode(neatTrainer.getBestGenome());

            validationErrorAfter = network.calculateError(validationSet);

            // logger.debug(String.format("Iteration #%d\tvError = %.12f", train.getIteration(), validationErrorAfter));

        } while (validationErrorAfter < validationErrorBefore);

        neatTrainer.finishTraining();

        this.method = network;
    }

    @Override
    public void test(BasicMLDataSet testSet)
    {
        int hamCount = 0, hamCorrect = 0;
        int spamCount = 0, spamCorrect = 0;
        NEATNetwork network = (NEATNetwork) this.method;

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

        logger.info(String.format("%s @ %d\t%s\tHP: %.2f%% (%d/%d)\tSP: %.2f%% (%d/%d)\tGP: %.2f%%", "NEAT", seed,
                folder.replace(DataSetHelper.BASE_FOLDER, ""), hamPrecision, hamCorrect, hamCount, spamPrecision, spamCorrect, spamCount, precision));
    }
}
