import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.classifiers.trees.BFTree;
import weka.core.SelectedTag;
import weka.core.Instance;
import java.util.Enumeration;
import weka.classifiers.meta.AdaBoostM1;

import java.io.File;

/**
 * 
 */
public class adult {

    /**
     * main method, runs things
     * @param args [description]
     */
    public static void main(String[] args) throws Exception{
        //load training data
        CSVLoader trainingdata = new CSVLoader();
        trainingdata.setSource(new File("adult.data"));
        Instances data = trainingdata.getDataSet();
        data.setClassIndex(data.numAttributes()-1);

        //load testing data
        CSVLoader testingdata = new CSVLoader();
        testingdata.setSource(new File("adult.test"));
        Instances test = testingdata.getDataSet();
        test.setClassIndex(test.numAttributes()-1);

        BFTree classifier = new BFTree();
        classifier.setPruningStrategy(new SelectedTag(BFTree.PRUNING_POSTPRUNING, BFTree.TAGS_PRUNING));
        classifier.buildClassifier(data);
        //System.out.println(classifier.classifyInstance(test.firstInstance()));
        double sumDifferences = 0;
        int count = 0;
        // for (Enumeration<Instance> e = test.enumerateInstances(); e.hasMoreElements();) {
        //     Instance i = e.nextElement();
        //     //System.out.print(classifier.classifyInstance(i)+" ");
        //     //System.out.println(i.value(test.numAttributes()-1));
        //     sumDifferences += (classifier.classifyInstance(i) - i.value(test.numAttributes()-1)) * (classifier.classifyInstance(i) - i.value(test.numAttributes()-1));
        //     count++;
        //     //sum squared differences
        // }
        // System.out.println(count);
        // System.out.println(sumDifferences/count);

        count = 0;
        sumDifferences = 0;
        AdaBoostM1 clf = new AdaBoostM1();
        String[] options = {"-I 25", "-W weka.classifiers.trees.BFTree"};
        clf.setOptions(options);
        clf.buildClassifier(data);
        System.out.println(clf.getRevision());
        for (Enumeration<Instance> e = test.enumerateInstances(); e.hasMoreElements();) {
            Instance i = e.nextElement();
            sumDifferences += (clf.classifyInstance(i) - i.value(test.numAttributes()-1)) * (clf.classifyInstance(i) - i.value(test.numAttributes()-1));
            count++;
            //sum squared differences
        }
        System.out.println(count);
        System.out.println(sumDifferences/count);
    }
}