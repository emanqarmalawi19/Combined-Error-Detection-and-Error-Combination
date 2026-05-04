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
    original_length = len(input_bits)  
    rng = np.random.default_rng(seed)  

    pad_len = (-original_length) % k  
    if pad_len > 0:  
        input_bits = np.concatenate([input_bits, np.zeros(pad_len, dtype=np.uint8)])  

    received_payload_bits = []           
    total_blocks = len(input_bits) // k  
    ack_blocks = 0   
    nack_blocks = 0  

    print(f"Total image bits entering Hard-Dec CRC + HARQ: {original_length:,}")  
    print(f"Block payload size k = {k}")     
    print(f"Total blocks = {total_blocks}")  
    print(f"Eb/N0 = {ebn0_db} dB")           
    print(f"Max transmissions M = {M}")      

    printed_first_nack = False  

    for block_index in range(total_blocks):  
        u = input_bits[block_index * k:(block_index + 1) * k]  
        x = crc_encode(u)       
        code_rate = k / len(x)  

        rx_buffer = []        
        block_passed = False  
        last_u_hat = None     

        is_early_block = block_index < verbose_blocks  
        
        block_log = [] 

        block_log.append(f"\n========== Block {block_index + 1}/{total_blocks} ==========")

        for tx_num in range(1, M + 1):  
            tx_symbols = bpsk_mod(x)  
            rx_symbols = awgn_channel(tx_symbols, ebn0_db, code_rate, rng)  

            rx_buffer.append(rx_symbols)  

            combined_soft = np.sum(np.vstack(rx_buffer), axis=0)  
            x_hat = (combined_soft < 0).astype(np.uint8)  

            passed, u_hat = crc_check(x_hat)  
            last_u_hat = u_hat                

            if passed:  
                block_log.append("CRC check: PASS")  
                block_log.append(f"ACK: decoded correctly after {tx_num} transmissions.")  
                block_log.append("Final decision: ACK")  

                received_payload_bits.extend(u_hat)  
                ack_blocks += 1      
                block_passed = True  
                break  

            else:  
                block_log.append(f"Transmission {tx_num}: CRC check: FAIL")  
                block_log.append("NACK: retransmission requested.")          

        if not block_passed:  
            block_log.append(f"CRC check: FAIL after {M} transmissions.")  
            block_log.append("Final decision: NACK")                       

            received_payload_bits.extend(last_u_hat)  
            nack_blocks += 1  

        if is_early_block:
            print("\n".join(block_log))
        elif not block_passed and not printed_first_nack:
            print("\n--- FIRST NACK ENCOUNTERED ---")
            print("\n".join(block_log))
            printed_first_nack = True #true so we don't print any more NACKs

    received_payload_bits = np.array(received_payload_bits, dtype=np.uint8)  
    received_payload_bits = received_payload_bits[:original_length]          

    print("\n========== Hard-Decision CRC + HARQ SUMMARY ==========")  
    print(f"Total blocks: {total_blocks}")  
    print(f"ACK blocks: {ack_blocks}")      
    print(f"NACK blocks: {nack_blocks}")    

    return received_payload_bits


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