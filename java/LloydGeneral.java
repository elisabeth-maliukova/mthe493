package main;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.lang3.StringUtils;

/**
 * Class that holds implementation of General Lloyds Algorithm
 * Mainly used by other classes, can be run itself for specific applications
 */
public class LloydGeneral {

    // Reduces list to single value by adding all terms
    private static double sum(List<Double> d) {
        return d.stream().reduce(0d, (a, b) -> a + b);
    }

    private static int[] toBin(int i, int width){
        String padded = StringUtils.leftPad(Integer.toBinaryString(i), width, '0');
        return padded.chars().map(Character::getNumericValue).toArray();
    }

    private static double[] linspace(double start, double stop, int num) {
        double[] array = new double[num];
        double step = (stop - start) / (num - 1);
        for (int i = 0; i < num; i++) {
            array[i] = start + i * step;
        }
        return array;
    }

    private static int mod2Add(int a, int b) {
        return (a + b) % 2;
    }

    public static double codeRate(int cbLength) {
        return Math.ceil(Math.log(cbLength) / Math.log(2));
    }

    // Optimized using Brian-Kernighan's Algorithm
    private static int hammingDistance(int x, int y) {
        int xor = x ^ y;
        int count = 0;
        while (xor != 0) {
            xor &= (xor - 1);
            count++;
        }
        return count;
    }

    public static double condProbBSC(int i, int j, double err, double cr) {
        int dH = hammingDistance(i, j);
        return Math.pow(err, dH) * Math.pow(1 - err, cr - dH);
    }

    private static double condProbPolya(int[] iBits, int[] jBits, double epsilon, double delta) {
        double prob_of_z_process_at_1 = 0;
        double result = 1;
        ArrayList<Integer> e = new ArrayList<Integer>();
        e.add(0);

        for (int index = 1; index <= iBits.length; index++) {
       
            e.add(mod2Add(iBits[index-1], jBits[index-1]));

            if (index == 1) {
                prob_of_z_process_at_1 = (e.get(1) == 1) ? epsilon : (1 - epsilon);
                result = prob_of_z_process_at_1;
            } else {
                double term1 = Math.pow(((epsilon + e.get(index-1) * delta) / (1 + delta)), e.get(index));
                double term2 = Math.pow((((1 - epsilon) + ((1 - e.get(index-1)) * delta)) / (1 + delta)), (1 - e.get(index)));
                result *= term1 * term2;
            }
        }
        return result;
    }

    public static double calcDistortionForAllBins(List<List<Double>> bins, double[] centroids, int cbLength,
        int numSamples, double err, boolean channelTypeBSC, double delta) {

        double distortion = 0;
        for (int i = 0; i < cbLength; i++) {
            for (Double sample : bins.get(i)) {
                for (int j = 0; j < cbLength; j++) {
                    if (channelTypeBSC) {
                        distortion += condProbBSC(i, j, err, codeRate(cbLength))
                                * Math.pow((sample - centroids[j]), 2);
                    } else {
                        int[] iBits = toBin(i, (int) codeRate(cbLength));
                        int[] jBits = toBin(j, (int) codeRate(cbLength));
                        distortion += condProbPolya(iBits, jBits, err, delta) * Math.pow((sample - centroids[j]), 2);
                    }
                }
            }
        }
        return distortion / numSamples;
    }

    public static List<List<Double>> assignSamplesToBins(List<Double> samples, double[] centroids, int cbLength,
            double err, boolean channelTypeBSC, double delta) {

        List<List<Double>> bins = new ArrayList<>();
        for (int i = 0; i < cbLength; i++) {
            bins.add(new ArrayList<>());
        }

        for (Double sample : samples) {
            double distortion = -1;
            int index = 0;
            for (int i = 0; i < cbLength; i++) {
                double newDistortion = 0;
                for (int j = 0; j < cbLength; j++) {
                    if (channelTypeBSC) {
                        newDistortion += condProbBSC(i, j, err, codeRate(cbLength)) * Math.pow((sample - centroids[j]), 2);
                    } else {
                        int[] iBits = toBin(i, (int) codeRate(cbLength));
                        int[] jBits = toBin(j, (int) codeRate(cbLength));
                        newDistortion += condProbPolya(iBits, jBits, err, delta) * Math.pow((sample - centroids[j]), 2);
                    }
                }
                if (newDistortion < distortion || distortion == -1) {
                    distortion = newDistortion;
                    index = i;
                }
            }
            bins.get(index).add(sample);
        }
        return bins;
    }

    public static double[] calculateCentroids(List<List<Double>> bins, int cbLength, double err, int numSamples,
            boolean channelTypeBSC, double delta) {

        double[] centroids = new double[cbLength];
        for (int j = 0; j < cbLength; j++) {
            double numerator = 0;
            double denominator = 0;
            for (int i = 0; i < cbLength; i++) {
                if (channelTypeBSC) {
                    numerator += condProbBSC(i, j, err, codeRate(cbLength)) * sum(bins.get(i));
                    denominator += condProbBSC(i, j, err, codeRate(cbLength)) * bins.get(i).size();
                } else {
                    int[] iBits = toBin(i, (int) codeRate(cbLength));
                    int[] jBits = toBin(j, (int) codeRate(cbLength));
                    numerator += condProbPolya(iBits, jBits, err, delta) * sum(bins.get(i));
                    denominator += condProbPolya(iBits, jBits, err, delta) * bins.get(i).size();
                }
            }
            // Avoiding division by zero
            centroids[j] = denominator != 0 ? numerator / denominator : 0;
        }
        return centroids;
    }

    public static double[] generalLloydsAlgorithm(List<Double> samples, int numSamples, double err, int cbLength,
            boolean channelTypeBSC, double delta) {

        List<Double> distortion = new ArrayList<Double>();
        double[] centroids = linspace(-1, 1, cbLength);

        List<List<Double>> bins = assignSamplesToBins(samples, centroids, cbLength, err, channelTypeBSC, delta);
        distortion.add(calcDistortionForAllBins(bins, centroids, cbLength, numSamples, err, channelTypeBSC, delta));

        int i = 0;
        while (true) {
            i++;
            bins = assignSamplesToBins(samples, centroids, cbLength, err, channelTypeBSC, delta);
            centroids = calculateCentroids(bins, cbLength, err, numSamples, channelTypeBSC, delta);
            distortion.add(calcDistortionForAllBins(bins, centroids, cbLength, numSamples, err, channelTypeBSC, delta));
            if (Math.abs(distortion.get(distortion.size() - 1) - distortion.get(distortion.size() - 2))
                    / distortion.get(distortion.size() - 2) <= 0.01) {
                break;
            }
        }

        return centroids;

        // Just return centroids for use in generating quantizers, if you want distortion returned you can change this function
        // Alternatively you can define a data type that returns both values, depending on your use case

        //return distortion.get(distortion.size() - 1); // returning the last distortion value (final iterated distortion)
    }


    //Example generating centroids... main use of this class is to call generalLloydsAlgorithm elsewhere
    public static void main(String args[]) {
        // Training Parameters
        final int mu = 0;
        final int sigma = 1;
        final int num_samples = (int) Math.pow(10, 5);
        final double channel_error_probability = 0.01;
        final double delta = 0.4;
        int[] codebook_lengths = { 1, 2, 4, 8 };
        double[][] centroids = new double[4][];

        NormalDistribution normDist = new NormalDistribution(mu, sigma);
        List<Double> samples = Arrays.stream(normDist.sample(num_samples)).boxed().collect(Collectors.toList());

        //Also show how timing works for performance comparisons
        long startTime = System.currentTimeMillis();
        for (int i = 0; i < codebook_lengths.length; i++) {
            centroids[i] = generalLloydsAlgorithm(samples, num_samples, channel_error_probability,
                    codebook_lengths[i], false, delta);
            System.out.println("Trained codebook length " + i);
        }
        long endTime = System.currentTimeMillis();
        System.out.println((endTime - startTime) / 1000.0);

        //Print out results centroids
        for (int i = 0; i < codebook_lengths.length; i++) {
            System.out.println("Centroids for codebook length " + codebook_lengths[i] + ":");
            for(int j = 0; j < centroids[i].length; j++)
                System.out.println(centroids[i][j] + " ");
        }
    }
}