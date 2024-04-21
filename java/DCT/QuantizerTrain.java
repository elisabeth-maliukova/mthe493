package main.DCT;

import static main.LloydGeneral.generalLloydsAlgorithm;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.math3.distribution.LaplaceDistribution;
import org.apache.commons.math3.distribution.NormalDistribution;

/**
 * Class that trains quantizers for specified paramaters and saves java object for later use
 */
public class QuantizerTrain {
    
    public static double[][] trainQuantizers(List<Double> source, List<Integer> codeRates, double err, int numSamples, boolean channelBSC, double delta) {
        
        double[][] quantizers = new double[9][];

        System.out.println("Starting quantizer training");
        for (int i = 1; i < 9; i++) {
          if (codeRates.contains(i)) {
                System.out.println("Starting rate " + i);
                int cbLength = (int) Math.pow(2, i);
                double[] result = generalLloydsAlgorithm(source, numSamples, err, cbLength, channelBSC, delta);
                quantizers[i] = result;
            } 
        }
        System.out.println("Finished quantizer training");
        return quantizers;
    }

    public static List<double[][]> createQuantizers(double errs, boolean channelBSC, double delta, int n, double mean, double variance) {
        // Create Zero mean, Unit variance Laplace and normal sources.
        List<Double> normSamples = new ArrayList<>();
        List<Double> laplSamples = new ArrayList<>();

        NormalDistribution normalDistribution = new NormalDistribution(mean, Math.sqrt(variance));
        LaplaceDistribution laplaceDistribution = new LaplaceDistribution(mean, Math.sqrt(variance));

        normSamples.addAll(Arrays.asList(Arrays.stream(normalDistribution.sample(n)).boxed().toArray(Double[]::new)));
        laplSamples.addAll(Arrays.asList(Arrays.stream(laplaceDistribution.sample(n)).boxed().toArray(Double[]::new)));

        List<Integer> dcRates = ChannelHelpers.getDCRates();
        List<Integer> acRates = ChannelHelpers.getACRates();

        // Train Quantizers for laplace and normal sources
        double[][] standardNormalQuantizers = trainQuantizers(normSamples, dcRates, errs, n, channelBSC, delta);
        double[][] standardLaplaceQuantizers = trainQuantizers(laplSamples, acRates, errs, n, channelBSC, delta);

        List<double[][]> result = new ArrayList<>();
        result.add(standardNormalQuantizers);
        result.add(standardLaplaceQuantizers);

        return result;
    }
    
    //Rewrite main to train whatever the quantizers you want
    //Example left in trains polya quantizers for 4 different errs and 3 deltas (12 total)
    public static void main(String[] args) throws FileNotFoundException, IOException{
        //Parameters to train to files
        int numSamples = 1000;
        boolean channelBSC = false;
        double mean = 0;
        double variance = 1;

        //Just for filename: to change bit mask being used, update active allocation in ChannelHelpers
        String bitMaskStr = "76BPB";

        double[] errsList = {0, 0.005, 0.01, 0.05};
        double[] deltaList = {0, 5, 10};
        for(int i = 0; i < 1; i++){
            for(int j = 0; j < 3; j++){
                //Not entirely updated by parameters in this example, change to whatever you want object file name to be...
                String fileName = "src/main/DCT/TrainedQObjs/" + "Polya(" + (int)Math.round(deltaList[j]) + ")-" + bitMaskStr + "-" + (int)(errsList[i]*1000) + "ept-" + (int)(numSamples/1000) + "k.dat";
                List<double[][]> quantizers = createQuantizers(errsList[i], channelBSC, deltaList[j], numSamples, mean, variance);

                //Flattens to primative array because its very easy to save and load primitive data without defining a data structure
                double[][][] flatQuant = new double[][][]{quantizers.get(0), quantizers.get(1)};

                //Sample output of primitive object to .dat file
                FileOutputStream fos = new FileOutputStream(fileName);
                ObjectOutputStream oos = new ObjectOutputStream(fos);
                oos.writeObject(flatQuant);
                oos.close();
                fos.close();
            }
        }
    }
}
