package main.DCT;

import static main.LloydGeneral.generalLloydsAlgorithm;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.math3.distribution.LaplaceDistribution;

/**
 * Class converts saved quantizers (.dat files) to text file form
 * These act as storage to pass quantizer data to python file
 * Uses raw text delimited by spaces and newlines (\n) as format
 * Can also use serialized JSON but since the quantizers are all in seperate files this was easier to throw together
 */
public class QuantizerConvert {

        //Only trained 76 bit mask quantizers (time), reused most data for other bit masks but trained some new AC rates
        //Left in bunch of example conversions you can rewrite to whatever use you need

        //Use DC component from 76 mask, train new AC component of rate 8
        public static void conv76to24() throws FileNotFoundException, IOException, ClassNotFoundException{
            //Define whatever directories in your project you want for files I/O
            String srcDir = "src/main/DCT/TrainedQObjs/";
            String destDir = "src/main/DCT/TrainedQTexts/";
            String channelStr = "";
            String paramStr = "";
            double[] errs = {0, 0.005, 0.01, 0.05};
            double[] deltas = {0, 5, 10};

            for(int errIndex = 0; errIndex < 4; errIndex++){ 
                for(int deltaIndex = 0; deltaIndex < 3; deltaIndex++){
                        channelStr = "Polya(" + (int)(deltas[deltaIndex]) + ")-";
                        paramStr = "-" + (int)(errs[errIndex] * 1000) + "ept-1k";

                        //Read in quantizers from .dat file
                        FileInputStream fis = new FileInputStream(srcDir + channelStr + "76BPB" + paramStr + ".dat");
                        ObjectInputStream ois = new ObjectInputStream(fis);
                        double[][][] quantizers = (double[][][]) ois.readObject();

                        FileOutputStream fos = new FileOutputStream(destDir + channelStr + "24BPB" + paramStr + ".txt", true);

                        //Fill in blank DC values
                        for(int i = 1; i < 8; i++){
                            fos.write("0\n".getBytes());
                        }
                        //Fill in DC value of rate 8
                        for(int l = 0; l < quantizers[0][8].length; l++)
                            fos.write((Double.toString(quantizers[0][8][l]) + " ").getBytes());
                        fos.write("\n".getBytes());
                        //Fill in blank AC values
                        for(int i = 1; i < 8; i++){
                            fos.write("0\n".getBytes());
                        }
                        //Train AC rate of 8
                        List<Double> laplSamples = new ArrayList<>();
                        LaplaceDistribution laplaceDistribution = new LaplaceDistribution(0, Math.sqrt(1));
                        laplSamples.addAll(Arrays.asList(Arrays.stream(laplaceDistribution.sample(1000)).boxed().toArray(Double[]::new)));
                        System.out.println("Training " + errs[errIndex] + " " + deltas[deltaIndex]);
                        double[] result = generalLloydsAlgorithm(laplSamples, 1000, errs[errIndex], 256, false, deltas[deltaIndex]);

                        //One line for each DC / AC rate up to max rate
                        //Each value for consecutive output points seperated by space
                        for(int l = 0; l < result.length; l++)
                            fos.write((Double.toString(result[l]) + " ").getBytes());
                        fos.write("\n".getBytes());

                        ois.close();
                        fos.close();
                }
            }

        }

        //All data for 58 bit mask in 76, just load in and convert
        public static void conv76to58() throws FileNotFoundException, IOException, ClassNotFoundException{
            //Define whatever directories in your project you want for files I/O
            String srcDir = "src/main/DCT/TrainedQObjs/";
            String destDir = "src/main/DCT/TrainedQTexts/";
            String channelStr = "Polya(10)-";
            String paramStr = "-50ept-1k";

            //Read in quantizers from .dat file
            FileInputStream fis = new FileInputStream(srcDir + channelStr + "76BPB" + paramStr + ".dat");
            ObjectInputStream ois = new ObjectInputStream(fis);
            double[][][] quantizers = (double[][][]) ois.readObject();

            FileOutputStream fos = new FileOutputStream(destDir + channelStr + "58BPB" + paramStr + ".txt", true);

            //Fill in blank DC values
            for(int i = 1; i < 8; i++){
                fos.write("0\n".getBytes());
            }
            //Fill in DC value of rate 8
            for(int l = 0; l < quantizers[0][8].length; l++)
                fos.write((Double.toString(quantizers[0][8][l]) + " ").getBytes());
            fos.write("\n".getBytes());
            //Fill in blank AC values
            for(int i = 1; i < 4; i++){
                fos.write("0\n".getBytes());
            }
            //Fill in AC values (4, 5, 6, 7)
            for(int i = 4; i < 8; i++){
                for(int j = 0; j < quantizers[1][i].length; j++)
                    fos.write((Double.toString(quantizers[1][i][j]) + " ").getBytes());
                fos.write("\n".getBytes());
            }
            fos.write("0\n".getBytes());

            ois.close();
            fos.close();
        }
    
        //Main used to convert 
        public static void main(String[] args) throws FileNotFoundException, IOException, ClassNotFoundException{
        
        //Add calls to additional conversion functions you want
        //conv76to24();
        //conv76to58();

        //Define whatever directories in your project you want for files I/O
        String srcDir = "src/main/DCT/TrainedQObjs/";
        String destDir = "src/main/DCT/TrainedQTexts/";
        //File name not including the file type
        String fileName = "Polya(10)-76BPB-0ept-1k";

        //Read in quantizers from .dat file
        FileInputStream fis = new FileInputStream(srcDir + fileName + ".dat");
        ObjectInputStream ois = new ObjectInputStream(fis);
        double[][][] quantizers = (double[][][]) ois.readObject();
        

        FileOutputStream fos = new FileOutputStream(destDir + fileName + ".txt", true);
        //Hard coded max rate of 8, if you want to expand bit masks just edit loops or write dynamically
        for(int i = 0; i < 2; i++){
            for(int j = 1; j < 9; j++){
                //Rate not in mask, write 0 and continue
                if(quantizers[i][j] == null){
                    fos.write("0\n".getBytes());
                    continue;
                }

                //One line for each DC / AC rate up to max rate
                //Each value for consecutive output points seperated by space (preserves order for encoding)
                for(int l = 0; l < quantizers[i][j].length; l++)
                    fos.write((Double.toString(quantizers[i][j][l]) + " ").getBytes());

                fos.write("\n".getBytes());
            }
        }

        ois.close();
        fos.close();
    }
}
