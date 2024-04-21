package main.DCT;

import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;

/**
 * Class holding functions related to image operations.
 */
public class ImageHelpers {

    public static BufferedImage resizeImage(BufferedImage originalImage, int targetWidth, int targetHeight) {
        BufferedImage resizedImage = new BufferedImage(targetWidth, targetHeight, BufferedImage.TYPE_INT_RGB);
        Graphics2D graphics2D = resizedImage.createGraphics();
        graphics2D.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        graphics2D.drawImage(originalImage, 0, 0, targetWidth, targetHeight, null);
        graphics2D.dispose();
        return resizedImage;
    }
    
    public static int[][] translateImage(int[][] testImage, int shift) {
        int[][] meanAdjustedImage = new int[testImage.length][testImage[0].length];
        for (int i = 0; i < testImage.length; i++) {
            for (int j = 0; j < testImage[0].length; j++) {
                meanAdjustedImage[i][j] = testImage[i][j] + shift;
                // If pixel translation is higher than 256, then bound it by 256
                if (meanAdjustedImage[i][j] > 256) {
                    meanAdjustedImage[i][j] = 256; 
                }
            }
        }
        return meanAdjustedImage;
    }

    public static int[][][] partitionImage(int[][] translatedImage) {
        List<int[][]> partitionedImage = new ArrayList<>();

        // Move 8 by 8 square over image
        for (int x = 0; x < translatedImage[0].length / 8; x++) {
            for (int y = 0; y < translatedImage.length / 8; y++) {
                int[][] dctSquare = new int[8][8];

                // Save each pixel in 8 by 8 square
                for (int i = 0; i < 8; i++) {
                    for (int j = 0; j < 8; j++) {
                        dctSquare[i][j] = translatedImage[i + (x * 8)][j + y * 8];
                    }
                }
                partitionedImage.add(dctSquare);
            }
        }

        int[][][] ret = new int[8][8][partitionedImage.size()];
        for(int block = 0; block < partitionedImage.size(); block++){
            for(int i = 0; i < 8; i++){
                for(int j = 0; j < 8; j++)
                    ret[i][j][block] = partitionedImage.get(block)[i][j];
            }
        }
        return ret;
    }

    public static int[][] reconstructImage(int[][][] inverseDCTTransform) {
        List<Integer> reconstructedImage = new ArrayList<>();

        for (int y = 0; y < 32; y++) {
            for (int i = 0; i < 8; i++) {
                for (int x = 0; x < 32; x++) {
                    for (int j = 0; j < 8; j++) {
                        reconstructedImage.add(inverseDCTTransform[i][j][x + 32 * y]);
                    }
                }
            }
        }

        int[][] reconstructedImageArray = new int[256][256];
        for (int i = 0; i < 256; i++) {
            for (int j = 0; j < 256; j++) {
                reconstructedImageArray[i][j] = reconstructedImage.get(i * 256 + j);
            }
        }

        return reconstructedImageArray;
    }
}
