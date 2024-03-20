#include <stdio.h>
#include <math.h>
#include <stdlib.h>

int calculate_hamming_distance(int x, int y) {
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

double calculate_conditional_probability(int hamming_distance, double error_probability, int code_rate) {
    double no_error_prob = 1 - error_probability;
    int num_bits_undistorted = code_rate - hamming_distance;
    double cond_prob = pow(error_probability, (double) hamming_distance) * pow(no_error_prob, (double) num_bits_undistorted);

    return cond_prob;
}

void create_conditional_prob_arr(double arr[], double error_probability, int code_rate) {
    for (int ham_dist = 0; ham_dist <= code_rate; ham_dist++) {
        arr[ham_dist] = calculate_conditional_probability(ham_dist, error_probability, code_rate);
    }
}

int calc_code_rate(int codebook_length) {
    return ceil(log2(codebook_length));
}

double calc_distortion_for_all_bins(int num_samples, double * bins[], int samples_per_bin[], double centroids[], int codebook_length, double code_rate, double hamm_dist_to_cond_probs[]) {
    double distortion = 0;

    double prob_of_codeword_given_sample, error, squared_error;
    
    for (int bin_index = 0; bin_index < codebook_length; bin_index++) {
        for (int sample_index = 0; sample_index < samples_per_bin[bin_index]; sample_index++) {
            for (int centroid_index = 0; centroid_index < codebook_length; centroid_index++) {
                prob_of_codeword_given_sample = hamm_dist_to_cond_probs[calculate_hamming_distance(bin_index, centroid_index)];
                error = (*(bins[bin_index] + sample_index) - centroids[centroid_index]);
                squared_error = error * error;
                distortion +=  prob_of_codeword_given_sample * squared_error;
            }
        }
    }

    return distortion / num_samples;
}

double calculate_expected_distortion_of_sample(double sample, int quantized_centroid_index, double centroids[], int codebook_length, double hamm_distances_to_cond_probs[]) {
    double distortion = 0;
    int ham_dist;
    double error, squared_error;
    for (int code = 0; code < codebook_length; code++) {
        ham_dist = calculate_hamming_distance(quantized_centroid_index, code);
        error = abs(sample - centroids[code]);
        squared_error = error * error;
        distortion += hamm_distances_to_cond_probs[ham_dist] * squared_error;
    }
    return distortion;
}

int find_centroid_index_of_sample_using_lin(double sample, int codebook_length, double centroids[], int index, int check_greater_centroids, double ham_dist_to_cond_probs[]) {
    double min_distortion, max_distortion;
    int min_index, max_index = index;
    min_distortion = max_distortion = calculate_expected_distortion_of_sample(sample, index, centroids, codebook_length, ham_dist_to_cond_probs);
    if (check_greater_centroids) {
        double new_distortion;
        for (index; index < codebook_length; index++) {
            new_distortion = calculate_expected_distortion_of_sample(sample, index, centroids, codebook_length, ham_dist_to_cond_probs);
            if (new_distortion < min_distortion) {
                min_distortion = new_distortion;
                min_index--
            } else {
                return min_index
            }
        }
    } else {
        double new_distortion;
        for (index; index > -1; index--) {
            new_distortion = calculate_expected_distortion_of_sample(sample, index, centroids, codebook_length, ham_dist_to_cond_probs);
            if (new_distortion < max_distortion) {
                max_distortion = new_distortion;
                max_index++
            } else {
                return max_index
            }
        }
    }
}

int find_centroid_index_of_sample_using_binary_search(double sample, int codebook_length, double centroids[], int min_index, int max_index, double ham_dist_to_cond_probs[]) {
    double distrotion_of_max_centroid, distortion_of_min_centroid;
    distortion_of_min_centroid = calculate_expected_distortion_of_sample(sample, min_index, centroids, codebook_length, ham_dist_to_cond_probs);
    distrotion_of_max_centroid = calculate_expected_distortion_of_sample(sample, max_index, centroids, codebook_length, ham_dist_to_cond_probs);

    int min_index_smaller = (distortion_of_min_centroid < distrotion_of_max_centroid);

    if (max_index - min_index == 1) {
        if (min_index_smaller) {
            if (min_index == 0){
                return min_index;
            } else if (calculate_expected_distortion_of_sample(sample, min_index - 1, centroids, codebook_length, calculate_hamming_distance) > distortion_of_min_centroid) {
                return min_index;
            } else{
                return find_centroid_index_of_sample_using_lin(sample, codebook_length, centroids, min_index - 1, 0);
            }
        } else {
            if (max_index == codebook_length - 1){
                return max_index;
            } else if (calculate_expected_distortion_of_sample(sample, max_index + 1, centroids, codebook_length, calculate_hamming_distance) > distortion_of_max_centroid) {
                return max_index;
            } else{
                return find_centroid_index_of_sample_using_lin(sample, codebook_length, centroids, max_index + 1, 1);
            }
        }
    }

    int midpoint_index = (max_index + min_index) / 2;

    if (min_index_smaller) {
        return find_centroid_index_of_sample_using_binary_search(sample, codebook_length, centroids, min_index, midpoint_index, ham_dist_to_cond_probs);
    } else {
        return find_centroid_index_of_sample_using_binary_search(sample, codebook_length, centroids, midpoint_index, max_index, ham_dist_to_cond_probs);
    }
}

void assign_samples_to_bin(int num_samples, double samples[], double centroids[], int codebook_length, double * bins[], int samples_per_bin[], double ham_dist_to_cond_probs[]) {
    int expected_samples_per_bin = num_samples/codebook_length;
    int elements_per_bin_array[codebook_length]; // allocated space in a bin containing samples where as samples_per_bin is the number of samples in that bin
    for (int bin = 0; bin < codebook_length; bin++) {
        free(bins[bin]);
        bins[bin] = (double *)calloc(expected_samples_per_bin, sizeof(double));
        elements_per_bin_array[bin] = expected_samples_per_bin;
        samples_per_bin[bin] = 0;
    }

    int bin_index = 0;
    double sample = 0;

    for (int sample_index = 0; sample_index < num_samples; sample_index++) {
        sample = samples[sample_index];
        bin_index = find_centroid_index_of_sample_using_binary_search(sample, codebook_length, centroids, 0, codebook_length - 1, ham_dist_to_cond_probs);
        if (samples_per_bin[bin_index] + 1 > elements_per_bin_array[bin_index]) {
            elements_per_bin_array[bin_index] += expected_samples_per_bin;
            bins[bin_index] = (double *)realloc(bins[bin_index], (elements_per_bin_array[bin_index]) * sizeof(double));
        }
        bins[bin_index][++(samples_per_bin[bin_index])] = sample;
    }
}

void calculate_centroids(double centroids[], int codebook_length, double * bins[], int samples_per_bin[], double ham_dist_to_cond_probs[]) {
    double sum;
    int sum_samples_per_bin[codebook_length];
    int sum;
    for (int bin = 0; bin < codebook_length; bin++) {
        sum = 0;
        for (int sample = 0; sample < samples_per_bin[bin]; sample++) {
            sum += bins[bin][sample];
        }
        sum_samples_per_bin[bin] = sum;
    }

    double numerator, denominator, cond_prob;
    for (int centroid_index = 0; centroid_index < codebook_length; centroid_index++) {
        numerator = denominator = 0;
        for (int output_centroid_index = 0; output_centroid_index < codebook_length; output_centroid_index++) {
            cond_prob = ham_dist_to_cond_probs[calculate_hamming_distance(centroid_index, output_centroid_index)];
            numerator += cond_prob * sum_samples_per_bin[output_centroid_index];
            denominator += cond_prob * samples_per_bin[output_centroid_index];
        }
        centroids[centroid_index] = numerator / denominator;
    }
}

void lloyds_algorithm(double samples[], int num_samples, double * bins[], double centroids[], double distortion[], double channel_error, double epsilon, int codebook_length) {
    int len_distotion = 100;
    distortion[] = (double *)calloc(len_distotion, sizeof(double));
    
    double initial_delta = 2 / (codebook_length - 1);
    for (int i = 0; i < codebook_length; i++) {
        centroids[i] = i * initial_delta - 1;
    }
    
    double code_rate = calc_code_rate(codebook_length);
    double hamm_dist_to_cond_prob_map[code_rate + 1];
    create_conditional_prob_arr(hamm_dist_to_cond_prob_map, channel_error, code_rate);

    int samples_per_bin;
    for (int bin = 0; bin < codebook_length; bin++) {
        bins[bin] = (double *)calloc(1, sizeof(double));
    }
    assign_samples_to_bin(num_samples, samples, centroids, codebook_length, bins, samples_per_bin, hamm_dist_to_cond_prob_map);

    distortion[0] = calc_distortion_for_all_bins(num_samples, bins, samples_per_bin, centroids, codebook_length, code_rate, hamm_dist_to_cond_prob_map);

    double delta_distortion = epsilon + 1;
    int count = 0;

    while (delta_distortion > epsilon) {
        count++;
        assign_samples_to_bin(num_samples, samples, centroids, codebook_length, bins, samples_per_bin, hamm_dist_to_cond_prob_map);
        calculate_centroids(centroids, codebook_length, bins, samples_per_bin, hamm_dist_to_cond_prob_map);

        if (count > len_distotion) {
            len_distotion += 100;
            distortion = (double *)realloc(distortion, len_distotion * sizeof(double));
        }
        distortion[count] = calc_distortion_for_all_bins(num_samples, bins, samples_per_bin, centroids, codebook_length, code_rate, hamm_dist_to_cond_prob_map);

        delta_distortion = abs(distortion[count] - distortion[count - 1]) / distortion[count - 1];

    }

}
