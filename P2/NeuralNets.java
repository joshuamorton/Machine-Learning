import dist.*;
import opt.*;
import opt.test.*;
import opt.example.*;
import opt.ga.*;
import shared.*;
import shared.ErrorMeasure;
import func.nn.backprop.*;

import java.util.*;
import java.io.*;
import java.text.*;

/**
 * Implementation of randomized hill climbing, simulated annealing, and genetic algorithm to
 * find optimal weights to a neural network that is classifying abalone as having either fewer 
 * or more than 15 rings. 
 *
 * @author Josh Morton
 * @version 1.0
 */
public class NeuralNets {
    private static Instance[] instances = initializeInstances("Hill_Valley_without_noise_Training.data");
    private static Instance[] testing = initializeInstances("Hill_Valley_without_noise_Testing.data");

    private static int inputLayer = 100, hiddenLayer = 5, outputLayer = 1, trainingIterations = 1000;
    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();
    
    private static ErrorMeasure measure = new SumOfSquaresError();

    private static DataSet set = new DataSet(instances);

    private static BackPropagationNetwork networks[] = new BackPropagationNetwork[3];
    private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[3];

    private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[3];
    private static String[] oaNames = {"RHC", "SA", "GA"};
    private static String results = "";

    private static DecimalFormat df = new DecimalFormat("0.000");

    public static void main(String[] args) {
        for(int i = 0; i < oa.length; i++) {
            networks[i] = factory.createClassificationNetwork(
                new int[] {inputLayer, hiddenLayer, outputLayer});
            nnop[i] = new NeuralNetworkOptimizationProblem(set, networks[i], measure);
        }

        oa[0] = new RandomizedHillClimbing(nnop[0]);
        oa[1] = new SimulatedAnnealing(1E11, .95, nnop[1]);
        oa[2] = new StandardGeneticAlgorithm(200, 100, 10, nnop[2]);

                
        String datafile = new String();
        for(int i = 0; i < oa.length; i++) {
            double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
            train(oa[i], networks[i], oaNames[i]); //trainer.train();
            end = System.nanoTime();
            trainingTime = end - start;
            trainingTime /= Math.pow(10,9);


            Instance optimalInstance = oa[i].getOptimal();
            networks[i].setWeights(optimalInstance.getData());


            double predicted, actual;
            start = System.nanoTime();
            for(int j = 0; j < testing.length; j++) {
                networks[i].setInputValues(testing[j].getData());
                networks[i].run();

                predicted = Double.parseDouble(testing[j].getLabel().toString());
                actual = Double.parseDouble(networks[i].getOutputValues().toString());
                //System.out.println(predicted + " " + actual);
                double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

            }
            end = System.nanoTime();
            testingTime = end - start;
            testingTime /= Math.pow(10,9);

            datafile += "\nResults for " + oaNames[i] + ": \nCorrectly classified " + correct + " instances." +
                        "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                        + df.format(correct/(correct+incorrect)*100) + "%\nTraining time: " + df.format(trainingTime)
                        + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";

            results +=  oaNames[i]+" ";
        }
        try {
            FileWriter outfile = new FileWriter("results.txt");
            outfile.write(datafile);
            outfile.close();
        } catch (Exception e) {

        }
        System.out.println(results);
    }

    private static void train(OptimizationAlgorithm oa, BackPropagationNetwork network, String oaName) {
        FileWriter outputFile;
        try {
            outputFile = new FileWriter(oaName+".txt");
        } catch (Exception e) {
            return;
        }
        //System.out.println("\nError results for " + oaName + "\n---------------------------");

        double length = 606;
        double incorrect;
        for(int i = 0; i < trainingIterations; i++) {
            incorrect = 0;
            oa.train();

            double error = 0;
            for(int j = 0; j < testing.length; j++) {
                network.setInputValues(testing[j].getData());
                network.run();

                Instance output = testing[j].getLabel(), example = new Instance(network.getOutputValues());
                example.setLabel(new Instance(Double.parseDouble(network.getOutputValues().toString())));
                error += measure.value(output, example);
                incorrect += Math.abs(Double.parseDouble(output.toString()) - Double.parseDouble(example.getLabel().toString())) < 0.5 ? 0 : 1;
            }
            try {
                outputFile.write(df.format(incorrect/length*100));
                outputFile.write("\n");
            } catch (Exception e) {

            }
            //System.out.println(df.format(error));
        }
        try {
            outputFile.close();
        } catch (Exception e) {

        }
    }

    private static Instance[] initializeInstances(String filename) {

        double[][][] attributes = new double[606][][];

        try {
            BufferedReader br = new BufferedReader(new FileReader(new File(filename)));

            for(int i = 0; i < attributes.length; i++) {
                Scanner scan = new Scanner(br.readLine());
                scan.useDelimiter(",");

                attributes[i] = new double[2][];
                attributes[i][0] = new double[100]; // 100 attributes
                attributes[i][1] = new double[1];

                for(int j = 0; j < 100; j++){
                    attributes[i][0][j] = Double.parseDouble(scan.next());
                }

                attributes[i][1][0] = Double.parseDouble(scan.next());
            }
        }
        catch(Exception e) {
            e.printStackTrace();
        }

        Instance[] data = new Instance[attributes.length];

        for(int i = 0; i < data.length; i++) {
            data[i] = new Instance(attributes[i][0]);
            data[i].setLabel(new Instance(attributes[i][1][0] < .5 ? 0 : 1));
        }

        return data;
    }
}

