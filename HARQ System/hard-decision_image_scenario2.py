import numpy as np      # used for arrays, bit operations, random noise, and BER calculation.
from PIL import Image   # used to open, crop, convert, save, and display the image.
import os               # used to check if the image path exists.

# Import JPEG encoding and decoding functions from our JPEG file.
from jpeg_colorful import encode_color_image_to_bits, decode_bits_to_color_image


# ============================================================
# CRC Functions
# ============================================================
# polynomial division in binary = division over GF(2)

# =======================================
# set the generator polynomial "Divisor"
# =======================================
CRC8_POLY = 0x2F  # This is a CRC-8, so the remainder has 8 bits; Therefore, the divisor has 8 + 1 = 9 bits; there is an invisible leading 1.
                  
def crc_encode(u):  
    reg = 0  # initialize the CRC register to zero.

    # =======================================
    # Division: augmented dataword ÷ divisor 
    # using shift register
    # =======================================
    for bit in u.astype(np.uint8):  # read each data bit one by one.
        msb = (reg >> 7) & 1        # get the most significant bit of the CRC register by shifting it 7 places to the right.
                                    # & 1: keeps only the last bit.
                                    # Save the leftmost bit of the CRC register before changing the register. 
       
        # =========================================
        # Bring down the next bit from the dataword
        # =========================================
        reg = ((reg << 1) & 0xFF) | int(bit)  # 1. shift the register left to make room on the right side for the new input bit.
                                              # 2. & 0xFF to keep only the lowest 8 bits.
                                              # 3. | OR: inserts the current input bit into the rightmost position.
        
        # =========================================
        # If leftmost bit is 1 → XOR with divisor
        # If leftmost bit is 0 → XOR with zeros
        # =========================================
        if msb:  # If the leftmost significant bit was 1
            reg ^= CRC8_POLY  # apply XOR with the CRC polynomial "Divisor".


    # =======================================
    # Append zeros to dataword
    # =======================================
    # Augmented dataword = original bits + 8 zeros

    for _ in range(8):  # add 8 zero bits to finish the CRC calculation.
        msb = (reg >> 7) & 1  
        reg = (reg << 1) & 0xFF  

        if msb:  
            reg ^= CRC8_POLY  

    crc_bits = np.array([(reg >> i) & 1 for i in range(7, -1, -1)], dtype=np.uint8)  # Convert the CRC value into 8 bits: the loop start at 7, stop before -1, move by -1 each time

    return np.concatenate([u.astype(np.uint8), crc_bits])  # Return the original data bits followed by CRC bits; Attach the remainder to the dataword to form the codeword


def crc_check(x_hat):  # Checks if a received block passes the CRC test.
    u_hat = x_hat[:-8].astype(np.uint8)      # Take all bits except the last 8 bits as data.
    crc_hat = x_hat[-8:].astype(np.uint8)    # Take the last 8 bits as the received CRC.
    recomputed_crc = crc_encode(u_hat)[-8:]  # Recalculate the CRC from the received data bits.

    passed = np.array_equal(crc_hat, recomputed_crc)  # Compare received CRC with recalculated CRC.

    return passed, u_hat  # Return CRC result and the received data bits without CRC.


# ============================================================
# Channel and Hard-Decision CRC + HARQ Functions
# ============================================================

def bpsk_mod(bits):  # Converts binary bits into BPSK symbols.
    return 1.0 - 2.0 * bits.astype(np.float64)  # Bit 0 becomes +1 and bit 1 becomes -1.


def awgn_channel(symbols, ebn0_db, code_rate=1.0, rng=None):  # Sends BPSK symbols through an AWGN noisy channel.
    if rng is None:                    # If no random generator is provided.
        rng = np.random.default_rng()  # Create a random number generator.

    ebn0_linear = 10 ** (ebn0_db / 10.0)            # Convert Eb/N0 "channel quality" from dB scale to linear scale.
    sigma2 = 1.0 / (2.0 * code_rate * ebn0_linear)  # Calculate the noise variance to control how strong the random noise is.
    noise = rng.normal(0.0, np.sqrt(sigma2), size=symbols.shape)  # Generate Gaussian noise.

    return symbols + noise  # Add noise to the transmitted symbols.


def hard_decision_harq_for_bitstream(input_bits, k=1024, M=3, ebn0_db=6.0, seed=7, verbose_blocks=10):  
    original_length = len(input_bits)  # Save original number of bits before padding
    rng = np.random.default_rng(seed)  # Create random generator for AWGN noise

    pad_len = (-original_length) % k  # Calculate how many zeros are needed to complete the last block
    if pad_len > 0:  # Check if padding is needed
        input_bits = np.concatenate([input_bits, np.zeros(pad_len, dtype=np.uint8)])  # Add zeros at the end

    received_payload_bits = []  # List to store decoded payload bits
    total_blocks = len(input_bits) // k  # Number of blocks after padding
    ack_blocks = 0  # Counter for blocks that pass CRC
    nack_blocks = 0  # Counter for blocks that fail after all transmissions

    print(f"Total image bits entering Hard-Dec CRC + HARQ: {original_length:,}")  # Print original bit length
    print(f"Block payload size k = {k}")     # Print block size
    print(f"Total blocks = {total_blocks}")  # Print number of blocks
    print(f"Eb/N0 = {ebn0_db} dB")           # Print channel Eb/N0 value
    print(f"Max transmissions M = {M}")      # Print maximum allowed transmissions

    printed_first_nack = False  # Used to print only the first NACK after early blocks

    for block_index in range(total_blocks):  # Loop through all blocks
        u = input_bits[block_index * k:(block_index + 1) * k]  # Extract current block of k bits
        x = crc_encode(u)  # Add CRC bits to the payload block
        code_rate = k / len(x)  # Calculate code rate = data bits / transmitted bits

        rx_buffer = []        # Store received signals from each transmission
        block_passed = False  # Flag to know if CRC passed for this block
        last_u_hat = None     # Store latest decoded payload estimate

        is_early_block = block_index < verbose_blocks  # True if this block should be printed in detail
        
        block_log = []  # Store log messages for this block

        block_log.append(f"\n========== Block {block_index + 1}/{total_blocks} ==========") 

        for tx_num in range(1, M + 1):  # Try transmitting this block up to M times
            tx_symbols = bpsk_mod(x)  # Convert bits to BPSK symbols
            rx_symbols = awgn_channel(tx_symbols, ebn0_db, code_rate, rng)  # Add AWGN noise to transmitted symbols

            rx_buffer.append(rx_symbols)  # Save received noisy symbols

            combined_soft = np.sum(np.vstack(rx_buffer), axis=0)  # Combine all retransmissions symbol by symbol
            x_hat = (combined_soft < 0).astype(np.uint8)  # Hard decision: negative becomes 1, positive becomes 0

            passed, u_hat = crc_check(x_hat)  # Check CRC and extract decoded payload bits
            last_u_hat = u_hat  # Save latest decoded payload bits

            if passed:  # If CRC passed
                block_log.append("CRC check: PASS")  # Log CRC success
                block_log.append(f"ACK: decoded correctly after {tx_num} transmissions.")  # Log ACK and transmission count
                block_log.append("Final decision: ACK")  # Log final ACK decision

                received_payload_bits.extend(u_hat)  # Add decoded payload bits to output
                ack_blocks += 1  # Increase ACK block counter
                block_passed = True  # Mark block as successfully decoded
                break  # Stop retransmissions for this block

            else:  # If CRC failed
                block_log.append(f"Transmission {tx_num}: CRC check: FAIL")  # Log failed transmission
                block_log.append("NACK: retransmission requested.")  # Log NACK request

        if not block_passed:  # If CRC did not pass after M transmissions
            block_log.append(f"CRC check: FAIL after {M} transmissions.")  # Log final CRC failure
            block_log.append("Final decision: NACK")  # Log final NACK decision

            received_payload_bits.extend(last_u_hat)  # Store the last decoded estimate even though CRC failed
                                                      # keeping the last decoded block even when CRC fails because JPEG decoder needs a complete bitstream to reconstruct the image.
            nack_blocks += 1  # Increase NACK block counter

        if is_early_block:  # If this is one of the first verbose blocks
            print("\n".join(block_log))  # Print detailed log for this block

        elif not block_passed and not printed_first_nack:  # If this is the first later NACK block
            print("\n--- FIRST NACK ENCOUNTERED ---")  # Print first NACK header
            print("\n".join(block_log))  # Print this failed block log
            printed_first_nack = True  # Prevent printing more later NACKs

    received_payload_bits = np.array(received_payload_bits, dtype=np.uint8)  # Convert output list to NumPy uint8 array
    received_payload_bits = received_payload_bits[:original_length]  # Remove padding bits and keep original length

    print("\n========== Hard-Decision CRC + HARQ SUMMARY ==========")  # Print summary header
    print(f"Total blocks: {total_blocks}")  # Print total processed blocks
    print(f"ACK blocks: {ack_blocks}")  # Print successful blocks
    print(f"NACK blocks: {nack_blocks}")  # Print failed blocks

    return received_payload_bits  # Return final recovered bitstream


# ============================================================
# Main Program
# ============================================================

if __name__ == "__main__": 

    image_path = r"F:\ECE 4 2nd semester\Signal Processing\image.jpg"  # the input image path.

    if not os.path.exists(image_path):  # Check if the image file doesn't exist.
        print(f"Error: Could not find '{image_path}'.")  # Print an error message if the image path is wrong.

    else:  # Continue if the image exists.

        img = Image.open(image_path).convert("YCbCr")  # Open the image and convert it to YCbCr color space.

        original_w, original_h = img.size  # Get the original image width and height.

        new_w = original_w - (original_w % 8)  # Make the width divisible by 8.
        new_h = original_h - (original_h % 8)  # Make the height divisible by 8.

        img_cropped = img.crop((0, 0, new_w, new_h))  # Crop the image so it can be divided into 8x8 blocks.

        print(f"Original image size: {original_w}x{original_h}")  # Print the original image size.
        print(f"Cropped image size: {new_w}x{new_h}")             # Print the cropped image size.

        print("\n1. Encoding color image to JPEG-style bitstream...")  # Print encoding step.

        image_bits = encode_color_image_to_bits(img_cropped)  # Encode the cropped YCbCr image into one bitstream.

        print(f"Generated image bitstream: {len(image_bits):,} bits")  # Print the number of generated bits.

        print("\n2. Sending image bitstream through hard-decision CRC + HARQ...") 

        received_bits = hard_decision_harq_for_bitstream(
            image_bits,  # JPEG-style image bitstream to be transmitted.
            k=1024,      # Number of data bits in each block before adding CRC.
            M=3,         # Maximum number of transmissions allowed for each block.
            ebn0_db=6.0, # Channel quality value in dB.
            seed=7,      # Random seed to make results repeatable.
            verbose_blocks=10  # Print detailed ACK/NACK information for the first 10 blocks.
        )

        bit_errors = np.sum(image_bits != received_bits)  # Count how many bits changed after transmission.
        ber = bit_errors / len(image_bits)                # Calculate the bit error rate.

        print(f"\nBit errors after Hard Decision CRC + HARQ: {bit_errors}")  # Print number of bit errors.
        print(f"BER after Hard Decision CRC + HARQ: {ber:.8f}")              # Print the bit error rate.

        print("\n3. Decoding received bitstream back to image...")  # Print decoding step.

        reconstructed_ycbcr = decode_bits_to_color_image(received_bits, new_h, new_w)  # Decode the received bits back into a YCbCr image.

        final_image = reconstructed_ycbcr.convert("RGB")  # Convert the reconstructed YCbCr image to RGB for display.

        output_filename = "reconstructed_color_image_CRC_harq.jpeg"  # Name of the output image file.

        final_image.save(output_filename)  # Save the final reconstructed image.

        print(f"\nDone! Saved output as '{output_filename}'.")  # Print success message.

        final_image.show()  # Display the final reconstructed image.
