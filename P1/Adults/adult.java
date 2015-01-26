import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
 
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

        //load testing data
        CSVLoader testingdata = new CSVLoader();
        testingdata.setSource(new File("adult.test"));
        Instances test = testingdata.getDataSet();
    }
    
}