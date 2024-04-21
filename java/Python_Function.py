# This file contains additional python functions used in addition to existing python code 
# to implement java quantizer training and PSNR calculations


# Read quantizer of passed name, hard coded max rate and source folder
def read_quant_from_file(path):

    folder = "quantizers"

    standard_laplace_quantizers = [[0], [0], [0], [0], [0], [0], [0], [0], [0]]
    standard_normal_quantizers = [[0], [0], [0], [0], [0], [0], [0], [0], [0]]

    line_count = 1
    with open(folder + "/" + path, 'r') as file:
        for line in file:
            raw_line = line.split()
            line_arr = [float(value) for value in raw_line]
            if line_count < 9:
                standard_normal_quantizers[line_count] = line_arr
            else:
                standard_laplace_quantizers[line_count - 8] = line_arr
            line_count = line_count + 1

    return [standard_normal_quantizers, standard_laplace_quantizers]



# Calculate PSNR from source and final image
def PSNR(img_orig, img_end):
    mse = np.mean((img_orig - img_end) ** 2)
    # In this case no noise, PSNR irrelevant
    if mse == 0:
        return 100
    psnr = 20 * math.log10(255.0 / math.sqrt(mse))
    return psnr




# Additional code used throughout DCT_Algorithm.py in final results.
# No complete example as it was pretty hacked together / rewritten in final days
# Everything you need to recreate results is in here


# Example importing quantizer from file
[standard_normal_quantizer, standard_laplace_quantizers] = read_quant_from_file("Polya(0)-76BPB-" + str(10) + "ept-1k.txt")

# Example ploting final image and printing PSNR
plt.imshow(final_image, cmap='gray')
plt.axis('off')
plt.savefig("output/test.jpg", bbox_inches='tight', pad_inches=0)
plt.clf()
print(str(PSNR(training_images[0], final_image)))

# Example getting compressed image and outputting PSNR
translated_image = translate_image(training_images[0], -128)
partitioned_image = partition_image(translated_image)
DCT_transform = DCT_transform_image(partitioned_image)
DCT_variances = get_DCT_variances(DCT_transform)
encoded_DCT_transform = encode_DCT_transform(standard_normal_quantizer, standard_laplace_quantizers, DCT_transform,DCT_variances)
trans_no_error = simulate_channel(encoded_DCT_transform, 0, channel_type, delta)
decode_no_error = decode_DCT_transform(trans_no_error, standard_normal_quantizer, standard_laplace_quantizers, DCT_variances)
inv_no_error = inverse_DCT_transform_image(decode_no_error)
recon_no_error = reconstruct_image(inv_no_error)
compressed_image = translate_image(recon_no_error, 128)
print(str(PSNR(training_images[0], compressed_image)))