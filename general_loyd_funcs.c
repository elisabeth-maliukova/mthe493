#include <stdio.h>
#include <math.h>
#include <stdlib.h>

int hamming_distance(int x, int y) {
    int diff_bits = x ^ y;
    int ham_dis = 0;
    int last_digit_dif_bit;

    while (diff_bits) {
        last_digit_dif_bit = diff_bits & 1;
        ham_dis += last_digit_dif_bit;
        diff_bits >> 1;
    }

    return ham_dis;
}

double conditional_probability(int input, int output, double error_probability, int code_rate) {
    double no_error_prob = 1 - error_probability;
    int num_bits_distorted = hamming_distance(input, output);
    int num_bits_undistorted = code_rate - num_bits_distorted;
    double cond_prob = pow(error_probability, (double) num_bits_distorted) * pow(no_error_prob, (double) num_bits_undistorted);

    return cond_prob;
}

double code_rate(int codebook_length) {
    return ceil(log2(codebook_length));
}

double calc_distortion_for_all_bins(int num_samples, double * bins[], int samples_per_bin[], double centroids[], int codebook_length, double code_rate, double channel_error_probability) {
    double distortion = 0;

    for (int bin_index = 0; bin_index < codebook_length; bin_index++) {
        for (int sample_index = 0; sample_index < samples_per_bin[bin_index]; sample_index++) {
            for (int centroid_index = 0; centroid_index < codebook_length; centroid_index++) {
                double prob_of_codeword_given_sample = conditional_probability(bin_index, centroid_index, channel_error_probability, code_rate);
                double error = (*(bins[bin_index] + (sample_index * sizeof(double))) - centroids[centroid_index]);
                double squared_error = error * error;
                distortion +=  prob_of_codeword_given_sample * squared_error;
            }
        }
    }

    return distortion / num_samples;
}

int find_centroid_index_of_sample_using_lin(double sample, int codebook_length, double centroids[], int index, int check_greater_centroids) {
    double min_distortion, max_distortion;
    int min_index, max_index = index;
    min_distortion = max_distortion = abs(sample - centroids[index]);
    if (check_greater_centroids) {
        double new_distortion;
        for (++index; index < codebook_length; index++) {
            new_distortion = abs(sample - centroids[index]);
            if (new_distortion < min_distortion) {
                min_distortion = new_distortion;
                min_index--
            } else {
                return min_index
            }
        }
    } else {
        double new_distortion;
        for (--index; index > -1; index--) {
            new_distortion = abs(sample - centroids[index]);
            if (new_distortion > max_distortion) {
                max_distortion = new_distortion;
                max_index++
            } else {
                return max_index
            }
        }
    }
}

int find_centroid_index_of_sample_using_bin(double sample, int codebook_length, double centroids[], int min_index, int max_index) {
    double distrotion_of_max_centroid, distortion_of_min_centroid;
    distortion_of_min_centroid = abs(sample - centroids[min_index]);
    distrotion_of_max_centroid = abs(sample - centroids[max_index]);

    int min_index_smaller = (distortion_of_min_centroid < distrotion_of_max_centroid);

    if (max_index - min_index == 1) {
        if (min_index_smaller) {
            if (min_index == 0){
                return min_index;
            } else if (abs(sample - centroids[min_index - 1] > distortion_of_min_centroid)) {
                return min_index;
            } else{
                return find_centroid_index_of_sample_using_lin(sample, codebook_length, centroids, min_index - 1, 0);
            }
        } else {
            if (min_index == 0){
                return max_index;
            } else if (abs(sample - centroids[axn_index + 1] > distortion_of_min_centroid)) {
                return max_index;
            } else{
                return find_centroid_index_of_sample_using_lin(sample, codebook_length, centroids, max_index + 1, 1);
            }
        }
    }

    int midpoint_index = (max_index + min_index) / 2;

    if (min_index_smaller) {
        return find_centroid_index_of_sample_using_bin(sample, codebook_length, centroids, min_index, midpoint_index);
    } else {
        return find_centroid_index_of_sample_using_bin(sample, codebook_length, centroids, midpoint_index, max_index);
    }
}

void assign_samples_to_bin(int num_samples, double samples[], double centroids[], int codebook_length, double * bins[], int samples_per_bin[]) {
    for (int bin = 0; bin < codebook_length; bin++) {
        samples_per_bin[bin] = 0;
    }

    int bin_index = 0;
    double sample = 0;

    for (int sample_index = 0; sample_index < num_samples; sample_index++) {
        sample = samples[sample_index];
        bin_index = find_centroid_index_of_sample_using_bin(sample, codebook_length, centroids, 0, codebook_length - 1);
        bins[bin_index] = realloc(bins[bin_index], ++(samples_per_bin[bin_index]) * sizeof(double));
        *(bins[bin_index] + samples_per_bin[bin_index] * sizeof(double)) = sample;
    }
}

void calculate_centroids(double centroids[], int codebook_length, double * bins[], int samples_per_bin[]) {
    double sum;
    for (int i = 0; i < codebook_length; i++) {
        sum = 0;
        for (int j = 0; j < samples_per_bin[i]; j++) {
            sum += *(bins[i] + j * sizeof(double));
        }
        centroids[i] = sum / samples_per_bin[i]
    }
}


