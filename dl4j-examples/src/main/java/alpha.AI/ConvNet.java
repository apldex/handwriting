package alpha.AI;

/**
 * Created by apldex on 9/26/16.
 */
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.*;
import java.io.*;
public class ConvNet {
    protected static final String [] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;

    public static void main(String[] args) throws Exception{
        int nRows = 30;
        int nEpoch = 10;
        int nChannels = 1;
        int nColumns = 30;
        int outputNum = 26;
        int batchSize = 11;
        int iterations = 1;
        int seed = 123;
        Random randNumGen = new Random(seed);

        DateFormat df = new SimpleDateFormat("dd/MM/yy HH:mm:ss");
        Date date = new Date();

        File file = new File("/Volumes/Data/HandwritingProject/Train Result .txt");

        FileWriter fw = new FileWriter((file.getAbsoluteFile()));
        BufferedWriter bw = new BufferedWriter(fw);

        System.out.println("Load dataset...");
        File parentDir = new File("/Volumes/Data/adrian/Datasets/Capt/");

        FileSplit filesInDir = new FileSplit(parentDir, allowedExtensions, randNumGen);

        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        BalancedPathFilter pathFilter = new BalancedPathFilter(randNumGen, allowedExtensions, labelMaker);


        InputSplit[] filesInDirSplit = filesInDir.sample(pathFilter, 80, 20);
        InputSplit trainData = filesInDirSplit[0];
        InputSplit testData = filesInDirSplit[1];

        ImageRecordReader trainReader = new ImageRecordReader(nRows,nColumns,1,labelMaker);
        ImageRecordReader testReader = new ImageRecordReader(nRows,nColumns,1,labelMaker);

        trainReader.initialize(trainData);
        testReader.initialize(testData);

        DataSetIterator trainIter = new RecordReaderDataSetIterator(trainReader, batchSize, 1, outputNum);
        DataSetIterator testIter = new RecordReaderDataSetIterator(testReader, batchSize, 1, outputNum);

        System.out.println("Build model...");

        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .iterations(iterations)
            .regularization(true).l2(0.0005)
            .learningRate(0.01)
            .weightInit(WeightInit.XAVIER)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(Updater.NESTEROVS).momentum(0.9)
            .list()
            .layer(0, new ConvolutionLayer.Builder(5, 5)
                //Convolution Layer #1
                .nIn(nChannels)
                .stride(1, 1)
                .nOut(100)
                .activation("identity")
                .build())
            .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2, 2)
                .stride(2, 2)
                .build())
            .layer(2, new ConvolutionLayer.Builder(5, 5)
                //Convolution Layer #2
                .stride(1, 1)
                .nOut(125)
                .activation("identity")
                .build())
            .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(4, 4)
                .stride(2, 2)
                .build())
            .layer(4, new DenseLayer.Builder().activation("relu")
                .nOut(500).build())
            .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nOut(outputNum)
                .activation("softmax")
                .build())
            .backprop(true).pretrain(false);
        new ConvolutionLayerSetup(builder,nRows,nColumns,1);

        MultiLayerConfiguration conf = builder.build();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);


        System.out.println("Train model...");
        model.setListeners(new ScoreIterationListener(1));

        for(int i=1; i<=5; i++){
            while(trainIter.hasNext()) {
                DataSet next = trainIter.next();
                next.binarize();
                INDArray feature = next.getFeatureMatrix();
                INDArray label = next.getLabels();
                model.fit(feature, label);
            }
            trainIter.reset();
            System.out.println("Complete epoch #"+i);

            System.out.println("Evaluate model...");
            Evaluation eval = new Evaluation();
            while(testIter.hasNext()){
                DataSet next = testIter.next();
                INDArray output = model.output(next.getFeatureMatrix(), false);
                eval.eval(next.getLabels(), output);
            }
            try{
                bw.write("Epoch #" + i);
                bw.newLine();
                bw.write(eval.stats());
                bw.newLine();
            } catch(Exception e){
                e.printStackTrace();
            }
            System.out.println(eval.stats());
            testIter.reset();
        }
        bw.close();
        System.out.println("====Training finished====");
    }
}

/*
 */

 /*  while(trainIter.hasNext()){
            DataSet next = trainIter.next();
            next.binarize();
            INDArray feature = next.getFeatureMatrix();
            INDArray label = next.getLabels();
            model.fit(feature,label);
        }

        System.out.println("Evaluate model..");
        Evaluation eval = new Evaluation();
        while(testIter.hasNext()){
            DataSet next = testIter.next();
            INDArray output = model.output(next.getFeatureMatrix(), false);
            eval.eval(next.getLabels(), output);
        }
        System.out.println(eval.stats());
        */
