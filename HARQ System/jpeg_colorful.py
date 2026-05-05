import numpy as np                   # used for arrays, matrices, and numerical calculations.
from scipy.fftpack import dct, idct  # to import Discrete Cosine Transform(DCT) and inverse DCT functions.
from PIL import Image                # used to open, convert, save, and display images.
import os                            # used to check whether the image file path exists.

# ============================================================
# 1. Quantization Matrices
# ============================================================
''' used in the compression step. They control how much information is kept or removed from each 8 x 8 DCT block.'''

Q_MATRIX = np.array([  
       # Luminance Quantization Matrix.
       # It is used for the Y channel, which represents brightness.
       # The human eye is very sensitive to brightness changes, so Q_MATRIX keeps more details compared to the color matrix.
        # Small numbers = keep more detail.
        # Large numbers = remove more detail.

    [16, 11, 10, 16, 24, 40, 51, 61],       
    [12, 12, 14, 19, 26, 58, 60, 55],       
    [14, 13, 16, 24, 40, 57, 69, 56],       
    [14, 17, 22, 29, 51, 87, 80, 62],       
    [18, 22, 37, 56, 68, 109, 103, 77],     
    [24, 35, 55, 64, 81, 104, 113, 92],     
    [49, 64, 78, 87, 103, 121, 120, 101],   
    [72, 92, 95, 98, 112, 100, 103, 99]     
])

C_MATRIX = np.array([  
       # Chrominance Quantization Matrix.
       # It is used for the Cb and Cr channels, which represent color information.
       # The human eye is less sensitive to small color changes, so C_MATRIX uses larger values and compresses color information more strongly."""

    [17, 18, 24, 47, 99, 99, 99, 99],  
    [18, 21, 26, 66, 99, 99, 99, 99],  
    [24, 26, 56, 99, 99, 99, 99, 99],  
    [47, 66, 99, 99, 99, 99, 99, 99],  
    [99, 99, 99, 99, 99, 99, 99, 99],  
    [99, 99, 99, 99, 99, 99, 99, 99],  
    [99, 99, 99, 99, 99, 99, 99, 99],  
    [99, 99, 99, 99, 99, 99, 99, 99]  
])


# ============================================================
# 2. Math Helper Functions
# ============================================================

def dct2(block):  # define a function to apply 2D DCT to one 8x8 image block.
    """Apply 2D Discrete Cosine Transform (DCT) on an 8x8 block.""" 
    return dct(dct(block.T, norm='ortho').T, norm='ortho')  # applies 2D DCT by transforming the block in both directions: rows and columns.
                                                            # .T swaps rows and columns
                                                            # norm='ortho' normalizes DCT to make reconstruction accurate by keeping the transform properly scaled.

def idct2(block):  # define a function to apply 2D inverse DCT to one block.
    """Apply 2D Inverse Discrete Cosine Transform (IDCT)."""  
    return idct(idct(block.T, norm='ortho').T, norm='ortho')  # Convert frequency coefficients back to pixel values.

def int_to_bits(value, num_bits=16):  # define a function to convert an integer into bits.
    """converts each quantized DCT coefficient into binary bits so it can be stored or transmitted as a bitstream."""  
    format_str = f'{{:0{num_bits}b}}'  # creates a format rule to write any number as binary using exactly num_bits bits, adding zeros at the beginning if needed.
    binary_str = format_str.format(int(value) & ((1 << num_bits) - 1))  # converts the coefficient to binary and uses num_bits bits to represent both positive and negative values.
                                                                        # shift <<: moves 1 left by num_bits places to create the bit range size.
                                                                        # subtract - 1: changes that size into a mask made of num_bits ones.
                                                                        # AND &: applies the mask to keep only the needed num_bits bits.
    return [int(b) for b in binary_str]  # Convert each binary character into integer 0 or 1.


def bits_to_int(bits, num_bits=16):  # define a function to convert bits back into a signed integer.
    """Convert an array of bits e.g. [1, 0, 1, 1] to a signed integer.""" 
    out = 0  # initialize the reconstructed integer.

    for bit in bits:  # loop through each bit.
        out = (out << 1) | int(bit)   # builds the binary number one bit at a time by shifting left the old bit and adding the current bit to the end.

    if out >= (1 << (num_bits - 1)):  # checks if the first bit shows a negative number.
        out -= (1 << num_bits)        # Converts the stored binary value into the correct negative number by subtracting (2^num_bits = (1 << num_bits)) from out.

    return out  # return the reconstructed signed integer.


# ============================================================
# 3. Channel Encoder / Decoder
# ============================================================

def encode_channel(channel_array, quant_matrix):  # Define a function to encode one image channel.
    """Encodes a single 2D color channel into a 1D bitstream."""  

    h, w = channel_array.shape  # Get the height and width of the channel.
    bitstream = []  # Create an empty list to store encoded bits.

    for i in range(0, h, 8):  # Move vertically through the channel in steps of 8 pixels.
        for j in range(0, w, 8):  # Move horizontally through the channel in steps of 8 pixels.

            block = channel_array[i:i + 8, j:j + 8]  # Extract one 8x8 block from the channel.
            centered_block = block.astype(np.float64) - 128.0  # converts pixel values to float so negative values are allowed and center pixel values around zero -128:127.
            dct_block = dct2(centered_block)  # Apply 2D DCT to convert the block to frequency domain.
            quantized_block = np.round(dct_block / quant_matrix)  # Quantize the DCT coefficients by division and rounding.
            flattened = quantized_block.flatten()  # Convert the 8x8 block into a 1D array of 64 coefficients "one long list".

            for coeff in flattened:  # Loop through every quantized coefficient.
                bits = int_to_bits(coeff, num_bits=16)  # Convert the coefficient into 16 bits.
                bitstream.extend(bits)  # Add the 16 bits to the total bitstream.

    return np.array(bitstream, dtype=np.uint8)  # Return the bitstream as a NumPy array of 0s and 1s.


def decode_channel(bitstream, h, w, quant_matrix):  # Define a function to decode one image channel.
    """Decodes a 1D bitstream back into a single 2D color channel."""  # Function description.

    reconstructed_channel = np.zeros((h, w), dtype=np.float64)  # Create an empty output channel.
    bit_idx = 0  # Initialize the current reading position in the bitstream.

    for i in range(0, h, 8):  # Move vertically through the output channel in steps of 8 pixels.
        for j in range(0, w, 8):  # Move horizontally through the output channel in steps of 8 pixels.

            block_coeffs = np.zeros((8, 8))  # Create an empty 8x8 block for decoded coefficients.

            for x in range(8):  # Loop through the rows of the 8x8 coefficient block.
                for y in range(8):  # Loop through the columns of the 8x8 coefficient block.

                    bits = bitstream[bit_idx:bit_idx + 16]  # Read the next 16 bits from the bitstream.
                    bit_idx += 16  # Move the bit index forward by 16 bits.
                    block_coeffs[x, y] = bits_to_int(bits, num_bits=16)  # Convert bits back into a signed coefficient.

            dequantized_block = block_coeffs * quant_matrix  # Reverse quantization by multiplying by the quantization matrix.
            idct_block = idct2(dequantized_block)  # Apply inverse DCT to return to spatial-domain pixels.
            reconstructed_block = idct_block + 128.0  # Add 128 to reverse the centering step.
            reconstructed_channel[i:i + 8, j:j + 8] = reconstructed_block  # Insert the reconstructed block into the channel.

    return np.clip(reconstructed_channel, 0, 255).astype(np.uint8)  # Clip values to 0-255 and convert to image pixel format.


# ============================================================
# 4. Image Encoding and Decoding processes
# ============================================================
def encode_color_image_to_bits(img_ycbcr):  # Encodes the full color image into one bitstream.
    y_img, cb_img, cr_img = img_ycbcr.split()  # Split the image into Y, Cb, and Cr channels.
    '''Y means luminance, or brightness: it stores how bright or dark each pixel is.
       Cb means blue chrominance: it stores color information related to the blue difference.
       Cr means red chrominance: it stores color information related to the red difference.'''

    y_arr = np.array(y_img)    # Convert the Y channel to a NumPy array.
    cb_arr = np.array(cb_img)  # Convert the Cb channel to a NumPy array.
    cr_arr = np.array(cr_img)  # Convert the Cr channel to a NumPy array.

    y_bits = encode_channel(y_arr, Q_MATRIX)    # Encode the Y brightness channel using the luminance matrix.
    cb_bits = encode_channel(cb_arr, C_MATRIX)  # Encode the Cb color channel using the chrominance matrix.
    cr_bits = encode_channel(cr_arr, C_MATRIX)  # Encode the Cr color channel using the chrominance matrix.

    return np.concatenate([y_bits, cb_bits, cr_bits])  # Combine all channel bits into one bitstream.


def decode_bits_to_color_image(bitstream, h, w):  # Decodes one bitstream back into a YCbCr image.
    blocks_per_channel = (h // 8) * (w // 8)  # Calculate number of 8x8 blocks in one channel.
    bits_per_channel = blocks_per_channel * 64 * 16   # Calculate bits per channel: blocks x 64 coefficients x 16 bits.

    rx_y_bits = bitstream[0:bits_per_channel]                          # Extract the Y-channel bits.
    rx_cb_bits = bitstream[bits_per_channel:2 * bits_per_channel]      # Extract the Cb-channel bits.
    rx_cr_bits = bitstream[2 * bits_per_channel:3 * bits_per_channel]  # Extract the Cr-channel bits.

    rec_y_arr = decode_channel(rx_y_bits, h, w, Q_MATRIX)    # Decode the Y channel.
    rec_cb_arr = decode_channel(rx_cb_bits, h, w, C_MATRIX)  # Decode the Cb channel.
    rec_cr_arr = decode_channel(rx_cr_bits, h, w, C_MATRIX)  # Decode the Cr channel.

    rec_y_img = Image.fromarray(rec_y_arr, mode='L')    # Convert reconstructed Y array to grayscale image.
    rec_cb_img = Image.fromarray(rec_cb_arr, mode='L')  # Convert reconstructed Cb array to grayscale image.
    rec_cr_img = Image.fromarray(rec_cr_arr, mode='L')  # Convert reconstructed Cr array to grayscale image.

    final_ycbcr = Image.merge('YCbCr', (rec_y_img, rec_cb_img, rec_cr_img))  # Merge Y, Cb, and Cr into one YCbCr image.

    return final_ycbcr  # Return reconstructed YCbCr image.

# ============================================================
# 5. Main Execution
# ============================================================
if __name__ == "__main__": 

    image_path = r"F:\ECE 4 2nd semester\Signal Processing\image.jpg"  #image path.

    if not os.path.exists(image_path):  # Check whether the image file doesn't exist.
        print(f"Error: Could not find '{image_path}'.")  # Print an error message if the image is missing.

    else:  # Continue if the image file exists.

        # ------------------------------
        # Transmitter Side = Encoding
        # ------------------------------

        img = Image.open(image_path).convert('RGB')  # Open the image and convert it to RGB.

        w, h = img.size  # Get the original image width and height.

        new_w = w - (w % 8)  # Make the width divisible by 8.
        new_h = h - (h % 8)  # Make the height divisible by 8.

        img_cropped = img.crop((0, 0, new_w, new_h))  # Crop the image so both dimensions are multiples of 8.

        img_ycbcr = img_cropped.convert('YCbCr')  # Convert the cropped image from RGB to YCbCr.

        print(f"Original size : {w}x{h}")  # Print the original image size.
        print(f"Cropped size  : {new_w}x{new_h}")  # Print the cropped image size.

        print("1. ENCODING color image to bitstream...")  # Print that encoding is starting.

        total_transmission_bits = encode_color_image_to_bits(img_ycbcr)  # Encode the full color image into one bitstream.
        
        # Save encoded bits to a text file
        encoded_bits_filename =  r"F:\ECE 4 2nd semester\Signal Processing\encoded_bits.txt"    # Create the name of the text file that will store the encoded bits.
        with open(encoded_bits_filename, "w") as file:                  # Open the text file in write mode; if it does not exist, Python will create it.
            file.write("".join(map(str, total_transmission_bits)))      # Convert all bits from numbers to strings, join them together, and write them to the file.

        print(f"Encoded bits saved to '{encoded_bits_filename}'.")      # Print a message to confirm that the encoded bits were saved successfully.

        print(f"Done! Total bits to transmit: {len(total_transmission_bits):,}")  # Print total number of transmitted bits.

        # ------------------------------
        # Receiver Side = Decoding
        # ------------------------------

        print("\n2. DECODING bitstream back to color image...")  # Print that decoding is starting.

        final_ycbcr = decode_bits_to_color_image(total_transmission_bits, new_h, new_w)  # Decode the bitstream back into a YCbCr image.

        final_rgb = final_ycbcr.convert('RGB')  # Convert the reconstructed image back to RGB.

        output_filename = "reconstructed_color_image.jpeg"  # Define the output image filename.

        final_rgb.save(output_filename)  # Save the reconstructed RGB image.

        print(f"Done! Saved output as '{output_filename}'.")  # Print confirmation that the image was saved.

        final_rgb.show()  # Display the reconstructed image.