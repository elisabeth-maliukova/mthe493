package main.DCT;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

import org.apache.commons.lang3.StringUtils;


/**
 * Class holding functions related to channel methods
 */
public class ChannelHelpers {

    //temporary table for testing purposes
    public static final int[][] BIT_ALLOCATION_76BPP_temp = 
                            {{4, 4, 4, 4, 4, 0, 0, 0},
                             {4, 4, 4, 4, 4, 0, 0, 0},
                             {4, 4, 4, 4, 4, 0, 0, 0},
                             {4, 4, 4, 4, 4, 0, 0, 0},
                             {4, 4, 4, 4, 4, 0, 0, 0},
                             {0, 0, 0, 0, 0, 0, 0, 0},
                             {0, 0, 0, 0, 0, 0, 0, 0},
                             {0, 0, 0, 0, 0, 0, 0, 0}};

    public static final int[][] BIT_ALLOCATION_76BPP = 
                            {{8, 7, 6, 4, 3, 0, 0, 0},
                             {7, 6, 5, 4, 0, 0, 0, 0},
                             {6, 5, 4, 0, 0, 0, 0, 0},
                             {4, 4, 0, 0, 0, 0, 0, 0},
                             {3, 0, 0, 0, 0, 0, 0, 0},
                             {0, 0, 0, 0, 0, 0, 0, 0},
                             {0, 0, 0, 0, 0, 0, 0, 0},
                             {0, 0, 0, 0, 0, 0, 0, 0}};
                        
    public static final int[][] BIT_ALLOCATION_58BPP = 
                            {{8, 7, 6, 4, 0, 0, 0, 0},
                             {7, 6, 5, 0, 0, 0, 0, 0},
                             {6, 5, 0, 0, 0, 0, 0, 0},
                             {4, 0, 0, 0, 0, 0, 0, 0},
                             {0, 0, 0, 0, 0, 0, 0, 0},
                             {0, 0, 0, 0, 0, 0, 0, 0},
                             {0, 0, 0, 0, 0, 0, 0, 0},
                             {0, 0, 0, 0, 0, 0, 0, 0}};

    public static final int[][] BIT_ALLOCATION_24BPP = 
                            {{8, 8, 0, 0, 0, 0, 0, 0},
                             {8, 0, 0, 0, 0, 0, 0, 0},
                             {0, 0, 0, 0, 0, 0, 0, 0},
                             {0, 0, 0, 0, 0, 0, 0, 0},
                             {0, 0, 0, 0, 0, 0, 0, 0},
                             {0, 0, 0, 0, 0, 0, 0, 0},
                             {0, 0, 0, 0, 0, 0, 0, 0},
                             {0, 0, 0, 0, 0, 0, 0, 0}};


    //CHANGE THIS VARIABLE TO UPDATE WHICH BIT TABLE USED IN ALL OTHER FILES
    public static int[][] activeAllocation = BIT_ALLOCATION_76BPP;
    
    public static List<Integer> getDCRates() {
        ArrayList<Integer> whyDidIDoThis = new ArrayList<Integer>();
        whyDidIDoThis.add(activeAllocation[0][0]);
        return whyDidIDoThis;
        //TODO fix this name lol
    }

    public static List<Integer> getACRates() {
        int[] flatMap = Arrays.stream(activeAllocation).flatMapToInt(Arrays::stream).toArray();
        flatMap[0] = 0;
        return new ArrayList<Integer>(Arrays.stream(flatMap).boxed().collect(Collectors.toSet()));
    }

    public static int transmitBSC (int n, double err, int rate){
        Random random = new Random();

        String bin = StringUtils.leftPad(Integer.toBinaryString(n), rate, '0');

        // Flip each bit with probability err
        StringBuilder output = new StringBuilder();
        for (char bit : bin.toCharArray()) {
            if (random.nextDouble() < err) {
                output.append((bit == '0') ? '1' : '0');
            } else {
                output.append(bit);
            }
        }
       
        return Integer.parseInt(output.toString(), 2);
    }


    public static double getPolyaTransitionProb(double prevState, double epsilon, double delta) {
        return (epsilon + (prevState * delta)) / (1 + delta);
    }

    public static int transmitPolya(int n, double epsilon, double delta, int rate) {

        String bin = StringUtils.leftPad(Integer.toBinaryString(n), rate, '0');

        BitSet distortedEncoding = new BitSet(bin.length());
        BitSet zProcess = new BitSet(bin.length());
        zProcess.set(0, Math.random() < epsilon);

        for (int index = 0; index < bin.length(); index++) {
            if (index == 0)
                distortedEncoding.set(0, bin.charAt(0) != '0' ^ zProcess.get(0));
            else {
                zProcess.set(index, Math.random() < getPolyaTransitionProb(zProcess.get(index - 1) ? 1 : 0, epsilon, delta));
                distortedEncoding.set(index, bin.charAt(index) != '0' ^ zProcess.get(index));
            }
        }

        int flippedNumber = 0;
        for (int i = 0; i < distortedEncoding.length(); i++) {
            if (distortedEncoding.get(i)) {
                flippedNumber |= (1 << (distortedEncoding.length() - i - 1));
            }
        }

        return flippedNumber;
    }


    public static int[][][] simulateChannel(int[][][] encodedValues, double err, boolean channelBSC, double delta) {
        int numBlocks = encodedValues[0][0].length;
        int[][][] sentValues = new int[8][8][numBlocks];

        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                int rate = activeAllocation[i][j];
                if (rate != 0) {
                    for (int k = 0; k < numBlocks; k++) {
                        if (channelBSC)
                            sentValues[i][j][k] = transmitBSC(encodedValues[i][j][k], err, rate);
                        else
                            sentValues[i][j][k] = transmitPolya(encodedValues[i][j][k], err, delta, rate);
                    }
                }
            }
        }

        return sentValues;
    }
}



