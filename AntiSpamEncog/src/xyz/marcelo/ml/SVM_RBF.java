package xyz.marcelo.ml;

import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.ml.svm.KernelType;
import org.encog.ml.svm.SVM;
import org.encog.ml.svm.SVMType;
import org.encog.ml.svm.training.SVMTrain;

import xyz.marcelo.common.MessageLabel;
import xyz.marcelo.helper.DataSetHelper;
import xyz.marcelo.helper.MethodHelper;

public class SVM_RBF extends AbstractClassifier
{
    @Override
    public void train(BasicMLDataSet trainingSet, BasicMLDataSet validationSet)
    {
        int inputCount = trainingSet.get(0).getInput().size();

        SVM svm = new SVM(inputCount, SVMType.SupportVectorClassification, KernelType.RadialBasisFunction);

        double bestC = 0, bestGamma = 0, bestError = Double.MAX_VALUE;

        for (int i = 1; i <= 1000; i++)
        {
            SVMTrain svmTrainTemp = new SVMTrain(svm, validationSet);
            svmTrainTemp.setC(Math.random());
            svmTrainTemp.setGamma(Math.random());
            svmTrainTemp.setFold((int) Math.log10(inputCount) + 1);
            svmTrainTemp.iteration();

            if (svmTrainTemp.getError() < bestError)
            {
                bestError = svmTrainTemp.getError();
                bestC = svmTrainTemp.getC();
                bestGamma = svmTrainTemp.getGamma();

                // logger.debug(String.format( "Best error so far = %f (i = %d, C=%f, GAMMA=%f)", bestError, i, bestC, bestGamma));
            }
        }

        SVMTrain svmTrain = new SVMTrain(svm, trainingSet);
        svmTrain.setC(bestC);
        svmTrain.setGamma(bestGamma);

        svmTrain.iteration();

        svmTrain.finishTraining();

        this.method = svm;
    }

    @Override
    public void test(BasicMLDataSet testSet)
    {
        int hamCount = 0, hamCorrect = 0;
        int spamCount = 0, spamCorrect = 0;
        SVM svm = (SVM) this.method;

        for (MLDataPair pair : testSet)
        {
            MLData input = pair.getInput();
            MLData ideal = pair.getIdeal();
            MLData output = svm.compute(input);

            if (MethodHelper.infer(ideal.getData()) == MessageLabel.HAM)
            {
                hamCount++;
                if (Math.abs(output.getData(0) - 0.0) < 1e-6)
                {
                    hamCorrect++;
                }
            }
            else if (MethodHelper.infer(ideal.getData()) == MessageLabel.SPAM)
            {
                spamCount++;
                if (Math.abs(output.getData(0) - 1.0) < 1e-6)
                {
                    spamCorrect++;
                }
            }
        }

        double hamPrecision = 100.0 * hamCorrect / hamCount;
        double spamPrecision = 100.0 * spamCorrect / spamCount;
        double precision = 100.0 * (hamCorrect + spamCorrect) / (hamCount + spamCount);

        logger.info(String.format("%s @ %d\t%s\tHP: %.2f%% (%d/%d)\tSP: %.2f%% (%d/%d)\tGP: %.2f%%", "SVC-RBF", seed,
                folder.replace(DataSetHelper.BASE_FOLDER, ""), hamPrecision, hamCorrect, hamCount, spamPrecision, spamCorrect, spamCount, precision));
    }
}
