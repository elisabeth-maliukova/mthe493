package main.DCT;

import static main.LloydGeneral.generalLloydsAlgorithm;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import java.awt.*;
import java.awt.image.DataBufferByte;

import org.apache.commons.math3.distribution.LaplaceDistribution;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.stat.descriptive.moment.Variance;

/**
 * Java implementation of the DCT algorithm, python used for final results in thesis
 */
public class DCT {

    //Values used for DCT and iDCT transform calculations
    private static double[][] c = new double[8][8];
    private static double[][] cT = new double[8][8];;


    //Helper method to initialize values needed for transforms
    private static void initDCTArrays(){
        for (int j = 0; j < 8; j++)
        {
            c[0][j]  = 1.0 / Math.sqrt(8.0);
            cT[j][0] = c[0][j];
        }

        for (int i = 1; i < 8; i++)
        {
            for (int j = 0; j < 8; j++)
            {
                double jj = (double)j;
                double ii = (double)i;
                c[i][j]  = Math.sqrt(2.0/8.0) * Math.cos(((2.0 * jj + 1.0) * ii * Math.PI) / (2.0 * 8.0));
                cT[j][i] = c[i][j];
            }
        }
    }


    //Manual implementation of 2D DCT transform
    public static double[][][] DCTTransformImage(int[][][] partitionedImage) {
        int numBlocks = partitionedImage[0][0].length;
        double[][][] dctTransform = new double[8][8][numBlocks];
        double temp[][] = new double[8][8];

        for (int block = 0; block < numBlocks; block++) {
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    temp[i][j] = 0.0;
                    for (int k = 0; k < 8; k++) {
                        temp[i][j] += (((partitionedImage[i][k][block]) - 128) * cT[k][j]);
                    }
                }
            }

            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    double temp1 = 0.0;
                    for (int k = 0; k < 8; k++) {
                        temp1 += (c[i][k] * temp[k][j]);
                    }
                    dctTransform[i][j][block] = temp1;
                }
            }
        }

        return dctTransform;
    }


    //Manual implementation of 2D iDCT transform 
    public static int[][][] inverseDCTTransformImage(double[][][] dctTransform) {
        int numBlocks = dctTransform[0][0].length;
        int invTransform[][][] = new int[8][8][numBlocks];
        double temp[][] = new double[8][8];

        for(int block = 0; block < numBlocks; block++) {
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    temp[i][j] = 0.0;
                    for (int k = 0; k < 8; k++) {
                        temp[i][j] += dctTransform[i][k][block] * c[k][j];
                    }
                }
            }

            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    double temp1 = 0.0;
                    for (int k = 0; k < 8; k++) {
                        temp1 += cT[i][k] * temp[k][j];
                    }

                    temp1 += 128;
                    invTransform[i][j][block] = (int)Math.round(temp1);

                    // IDK if this was needed, played with it but eventually used python to show DCT / iDCT so didn't matter

                    // if (temp1 < 0)
                    // {
                    //     invTransform[i][j][block] = 0;
                    // }
                    // else if (temp1 > 255)
                    // {
                    //     invTransform[i][j][block] = 255;
                    // }
                    // else
                    // {
                    //     invTransform[i][j][block] = (int)Math.round(temp1);
                    // }
                }
            }
        }

        return invTransform;
    }


    public static double[][] getDCTVariances(double[][][] DCTTransform) {
        double[][] DCTVariances = new double[8][8];

        Variance variance = new Variance();

        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                DCTVariances[i][j] = variance.evaluate(DCTTransform[i][j]);
            }
        }
        return DCTVariances;
    }


    public static double[][] trainQuantizers(List<Double> source, List<Integer> codeRates, double err, int numSamples, boolean channelBSC, double delta) {
        
        double[][] quantizers = new double[9][];

        System.out.println("Starting quantizer training");
        for (int i = 1; i < 9; i++) {
          if (codeRates.contains(i)) {
                System.out.println("Starting rate " + i);
                int cbLength = (int) Math.pow(2, i);
                double[] result = generalLloydsAlgorithm(source, numSamples, err, cbLength, channelBSC, delta);
                // Store quantizer at index i
                quantizers[i] = result;
            } 
        }
        System.out.println("Finished quantizer training");
        return quantizers;
    }


    public static List<double[][]> createQuantizers(double errs, boolean channelBSC, double delta) {
        int numSamples = 1000;
        double mean = 0;
        double variance = 1;

        // Create Zero mean, Unit variance Laplace and normal sources.
        List<Double> normSamples = new ArrayList<>();
        List<Double> laplSamples = new ArrayList<>();

        NormalDistribution normalDistribution = new NormalDistribution(mean, Math.sqrt(variance));
        LaplaceDistribution laplaceDistribution = new LaplaceDistribution(mean, Math.sqrt(variance));

        normSamples.addAll(Arrays.asList(Arrays.stream(normalDistribution.sample(numSamples)).boxed().toArray(Double[]::new)));
        laplSamples.addAll(Arrays.asList(Arrays.stream(laplaceDistribution.sample(numSamples)).boxed().toArray(Double[]::new)));

        List<Integer> dcRates = ChannelHelpers.getDCRates();
        List<Integer> acRates = ChannelHelpers.getACRates();

        // Train Quantizers for laplace and normal sources
        double[][] standardNormalQuantizers = trainQuantizers(normSamples, dcRates, errs, numSamples, channelBSC, delta);
        double[][] standardLaplaceQuantizers = trainQuantizers(laplSamples, acRates, errs, numSamples, channelBSC, delta);

        List<double[][]> result = new ArrayList<>();
        result.add(standardNormalQuantizers);
        result.add(standardLaplaceQuantizers);
        return result;
    }


    public static int[] encodeCoefficients(double[][] standardQuantizers, double[] coefficients, int rate) {
        int coefficientsLen = coefficients.length;
        int[] encodedCoefficients = new int[coefficientsLen];

        if (rate != 0) {
          int codebookLength = 1 << rate;
          for (int i = 0; i < coefficientsLen; i++) {
              double distortion = Double.POSITIVE_INFINITY;
              for (int j = 0; j < codebookLength; j++) {
                  if (Math.pow(standardQuantizers[rate][j] - coefficients[i], 2) < distortion) {
                      distortion = Math.pow(standardQuantizers[rate][j] - coefficients[i], 2);
                      encodedCoefficients[i] = j;
                  }
              }
          }
        } else {
          Arrays.fill(encodedCoefficients, 0);
        }

        return encodedCoefficients;
    }

    public static double[] decodeCoefficients(double[][] standardQuantizers, int[] encodings, int rate) {
        int coefficientsLen = encodings.length;
        double[] decodedCoefficients = new double[coefficientsLen];
        
        if (rate != 0) {
            for (int i = 0; i < coefficientsLen; i++)
                decodedCoefficients[i] = standardQuantizers[rate][encodings[i]];
        } else {
            Arrays.fill(decodedCoefficients, 0);
        }

        return decodedCoefficients;
    }


    public static int[][][] encodeDCTTransform(double[][] standardNormalQuantizers, double[][] standardLaplaceQuantizers, double[][][] DCTTransform, double[][] DCTVariances) {
        int numBlocks = DCTTransform[0][0].length;
        double[][][] varianceAdjustedValues = new double[8][8][numBlocks];
        int[][][] encodedValues = new int[8][8][numBlocks];

        // Adjust values for variances
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                for (int k = 0; k < numBlocks; k++) {
                    varianceAdjustedValues[i][j][k] = DCTTransform[i][j][k] / Math.sqrt(DCTVariances[i][j]);
                }
            }
        }

        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                if (i == 0 && j == 0)
                    encodedValues[i][j] = encodeCoefficients(standardNormalQuantizers, varianceAdjustedValues[0][0], ChannelHelpers.activeAllocation[0][0]);
                else
                    encodedValues[i][j] = encodeCoefficients(standardLaplaceQuantizers, varianceAdjustedValues[i][j], ChannelHelpers.activeAllocation[i][j]);
            }
        }

        return encodedValues;
    }

    public static double[][][] decodeDCTTransform(int[][][] transmittedValues, double[][] standardNormalQuantizers, double[][] standardLaplaceQuantizers, double[][] DCTVariances) { 
        int numBlocks = transmittedValues[0][0].length;
        double[][][] decodedValues = new double[8][8][numBlocks];

        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                if (i == 0 && j == 0)
                    decodedValues[i][j] = decodeCoefficients(standardNormalQuantizers, transmittedValues[0][0], ChannelHelpers.activeAllocation[0][0]);
                else
                    decodedValues[i][j] = decodeCoefficients(standardLaplaceQuantizers, transmittedValues[i][j], ChannelHelpers.activeAllocation[i][j]);
            }
        }

        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                for (int k = 0; k < numBlocks; k++)
                    decodedValues[i][j][k] *= Math.sqrt(DCTVariances[i][j]);
            }
        }

        return decodedValues;
    }


    //Method to convert integer array to displayable image
    private static BufferedImage intToBuffered(int[][] img){
      int width = img[0].length;
      int height = img.length;

      BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);

      byte[] data = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();

      for (int y = 0; y < height; y++) {
          for (int x = 0; x < width; x++) {
              int pixelValue = img[y][x];
              // Ensure the pixel value is within 0-255 range
              int clampedValue = Math.max(0, Math.min(255, pixelValue));
              data[y * width + x] = (byte) clampedValue;
          }
      }

      return image;
    }

    //Method to create window displaying all passed images
    public static void displayImages(BufferedImage... images) {
        JFrame frame = new JFrame();
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        JPanel panel = new JPanel(new GridLayout(1, images.length));
        for (BufferedImage image : images) {
            JLabel label = new JLabel(new ImageIcon(image));
            panel.add(label);
        }

        frame.getContentPane().add(panel, BorderLayout.CENTER);

        frame.pack();
        // Center the frame on the screen
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
    }

    // Example main function, this was rewritten many times for different uses,
    // Reuse example code to perform whatever funciton you need
    public static void main(String[] args) throws InterruptedException{ 
        
        //Initalize values for DCT calculations
        initDCTArrays();

        //Parameters
        boolean channelBSC = true;
        double channelError = 0;
        double delta = 0.4;

        //Image import and store, directory path relative to project root path
        String dir = "images/test";
        File[] imageFiles = new File(dir).listFiles((dir1, name) -> name.endsWith(".jpg"));
        List<int[][]> trainingImages = new ArrayList<>();
        if (imageFiles != null) {
            for (File imageFile : imageFiles) {
                try {
                    BufferedImage img = ImageHelpers.resizeImage(ImageIO.read(imageFile), 256, 256);
                    int width = img.getWidth();
                    int height = img.getHeight();
                    int[][] grayscaleImage = new int[height][width];
                    //Manual conversion to grayscale (java cv2 was not being nice)
                    for (int y = 0; y < height; y++) {
                        for (int x = 0; x < width; x++) {
                            int rgb = img.getRGB(x, y);
                            int r = (rgb >> 16) & 0xFF;
                            int g = (rgb >> 8) & 0xFF;
                            int b = rgb & 0xFF;
                            int gray = (int) (0.2126 * r + 0.7152 * g + 0.0722 * b);
                            grayscaleImage[y][x] = gray;
                        }
                    }
                    trainingImages.add(grayscaleImage);
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }

        //Creates quantizers from scratch, this was used before switching to python after quantizer training
        //If want to use complete java, can also reuse method from QuantizerConvert to load quantizers from .dat or .txt
        List<double[][]> quantizers = createQuantizers(channelError, channelBSC, delta);

        int[][] translatedImage = ImageHelpers.translateImage(trainingImages.get(0), -128);
        int[][][] partitionedImage = ImageHelpers.partitionImage(translatedImage);
        double[][][] dctTransform = DCTTransformImage(partitionedImage);
        double[][] DCTVariances = getDCTVariances(dctTransform);

        int[][][] encodedDCT = encodeDCTTransform(quantizers.get(0), quantizers.get(1), dctTransform, DCTVariances);
        int[][][] transDCT = ChannelHelpers.simulateChannel(encodedDCT, channelError, channelBSC, delta);
        double[][][] decodeDCT = decodeDCTTransform(transDCT, quantizers.get(0), quantizers.get(1), DCTVariances);

        int[][][] idctTransform = inverseDCTTransformImage(decodeDCT);
        int[][] reconstrucedImg = ImageHelpers.reconstructImage(idctTransform);
        int[][] finalImage = ImageHelpers.translateImage(reconstrucedImg, 128);
        BufferedImage finalBuffImage = intToBuffered(finalImage);

        //Get initial image
        BufferedImage grayInit = intToBuffered(trainingImages.get(0));

        //Plot Images
        displayImages(new BufferedImage[] { grayInit, finalBuffImage });
    }
}