import numpy as np
from PIL import Image
import os

# Import JPEG encoding and decoding functions from your local file
from jpeg_colorful import encode_color_image_to_bits, decode_bits_to_color_image

# ============================================================
# CRC Functions (Cyclic Redundancy Check)
# ============================================================
# This polynomial (0x2F) represents the mathematical divisor used to create the CRC checksum.
CRC8_POLY = 0x2F
                  
def crc_encode(u):  
    """
    Takes an array of bits, calculates an 8-bit checksum, and appends it to the end.
    This acts as a mathematical signature to detect errors later.
    """
    reg = 0  # Initialize a shift register to zero
    
    # Process the data bits one by one
    for bit in u.astype(np.uint8):  
        msb = (reg >> 7) & 1  # Get the Most Significant Bit of the register      
        reg = ((reg << 1) & 0xFF) | int(bit)  # Shift register left and pull in the new data bit
        if msb:  
            reg ^= CRC8_POLY  # Perform XOR division if the MSB was 1

    # Process 8 extra zeros to flush the register and get the final remainder
    for _ in range(8):  
        msb = (reg >> 7) & 1  
        reg = (reg << 1) & 0xFF  
        if msb:  
            reg ^= CRC8_POLY  

    # Extract the 8 bits of the remainder (the actual CRC checksum)
    crc_bits = np.array([(reg >> i) & 1 for i in range(7, -1, -1)], dtype=np.uint8)  
    
    # Combine the original data and the 8-bit checksum into one array
    return np.concatenate([u.astype(np.uint8), crc_bits])  

def crc_check(x_hat):  
    """
    Separates the data from the received checksum, recalculates the checksum 
    from the data, and compares the two to verify integrity.
    """
    u_hat = x_hat[:-8].astype(np.uint8)      # Extract all bits except the last 8 (the data)
    crc_hat = x_hat[-8:].astype(np.uint8)    # Extract only the last 8 bits (the received checksum)
    recomputed_crc = crc_encode(u_hat)[-8:]  # Recalculate checksum using our function
    
    # Check if the received checksum perfectly matches our newly calculated one
    passed = np.array_equal(crc_hat, recomputed_crc)  
    return passed, u_hat  

# ============================================================
# Channel Functions (Modulation and Noise)
# ============================================================
def bpsk_mod(bits):  
    """
    Converts binary 0s and 1s into electrical signal voltage levels (+1.0 and -1.0).
    Bit 0 becomes +1.0, Bit 1 becomes -1.0.
    """
    return 1.0 - 2.0 * bits.astype(np.float64)  

def awgn_channel(symbols, ebn0_db, code_rate=1.0, rng=None):  
    """
    Simulates the wireless air interface by adding random Gaussian noise to the signals.
    """
    if rng is None:                    
        rng = np.random.default_rng()  # Create a random number generator if none exists
        
    # Convert logarithmic signal-to-noise ratio (dB) to a linear scale
    ebn0_linear = 10 ** (ebn0_db / 10.0)            
    
    # Calculate the variance (power) of the noise based on the signal-to-noise ratio
    sigma2 = 1.0 / (2.0 * code_rate * ebn0_linear)  
    
    # Generate random noise values following a Gaussian (Normal) distribution
    noise = rng.normal(0.0, np.sqrt(sigma2), size=symbols.shape)  
    
    # Add the noise to the original transmitted symbols
    return symbols + noise  

# ============================================================  
# Randomized IR-HARQ Simulation (Convolutional Style)  
# ============================================================  

def convolutional_ir_harq(input_bits, k=1024, M=4, ebn0_db=6.0, seed=7, verbose_blocks=10):  
    original_length = len(input_bits)  # Store the original number of image bits before padding
    rng = np.random.default_rng(seed)  # Create a seeded random number generator for repeatable simulation results

    pad_len = (-original_length) % k  # Calculate how many zeros are needed so the bitstream divides evenly into blocks
    if pad_len > 0:                   # Check whether padding is required
        input_bits = np.concatenate([input_bits, np.zeros(pad_len, dtype=np.uint8)])  # Add zero bits to complete the final block

    received_payload_bits = []  # Create an empty list to store decoded payload bits from all blocks
    total_blocks = len(input_bits) // k  # Calculate the total number of fixed-size blocks
    ack_blocks = 0   # Initialize the counter for successfully decoded blocks
    nack_blocks = 0  # Initialize the counter for blocks that fail after all retransmissions

    print(f"Total image bits entering True IR-HARQ: {original_length:,}")  # Print the original number of bits entering the HARQ system
    print(f"Block payload size k = {k}")     # Print the number of payload bits per block
    print(f"Total blocks = {total_blocks}")  # Print the total number of blocks
    print(f"Eb/N0 = {ebn0_db} dB")           # Print the channel signal-to-noise ratio in dB
    print(f"Max transmissions M = {M}")      # Print the maximum number of transmissions allowed per block

    printed_first_nack = False  # used to track whether the first failed block has already been printed

    for block_index in range(total_blocks):  # Loop over each block in the padded input bitstream

        # 1. extract the specific 1024 bits for the current block 
        u = input_bits[block_index * k:(block_index + 1) * k]  

        # 2. Add an 8-bit CRC checksum to the current block
        u_crc = crc_encode(u) 

        # 3. MOTHER CODE GENERATION (creating parity using Convolutional logic)
        rv0 = u_crc  # Create redundancy version 0 as the original systematic data plus CRC
        rv1 = np.bitwise_xor(u_crc, np.roll(u_crc, 1))  # Create redundancy version 1 using XOR with a 1-bit circular shift
        rv2 = np.bitwise_xor(u_crc, np.roll(u_crc, 3))  # Create redundancy version 2 using XOR with a 3-bit circular shift
        rv3 = np.bitwise_xor(u_crc, np.roll(u_crc, 5))  # Create redundancy version 3 using XOR with a 5-bit circular shift

        mother_code_rvs = [rv0, rv1, rv2, rv3]  # Store all redundancy versions in a list
        rv_size = len(u_crc)  # Store the length of each redundancy version, including CRC bits

        # 4. RECEIVER INITIALIZATION
        rx_buffer_soft = np.zeros((M, rv_size), dtype=np.float64)  # Create a buffer to store received noisy symbols for each transmission
        block_passed = False  # Track whether the current block passes CRC
        last_u_hat = None  # Store the latest decoded payload estimate in case all transmissions fail

        is_early_block = block_index < verbose_blocks  # Decide whether to print detailed logs for this block

        block_log = []  # Create a list to store log messages for this block
        block_log.append(f"\n========== Block {block_index + 1}/{total_blocks} ==========")  # Add a block header to the log

        # 5. TRANSMISSION LOOP (up to M attempts allowed per block)
        for tx_num in range(M):  # Loop through transmission attempts up to the maximum allowed

            tx_bits = mother_code_rvs[tx_num]  # Select the redundancy version for the current transmission
            tx_symbols = bpsk_mod(tx_bits)  # Convert the selected bits into BPSK symbols
            rx_symbols = awgn_channel(tx_symbols, ebn0_db, code_rate=1.0, rng=rng)  # Pass the symbols through the noisy AWGN channel
            rx_buffer_soft[tx_num, :] = rx_symbols  # Store the received noisy symbols in the soft buffer

            #  6. DECODING & SOFT COMBINING 
            combined_energy = rx_buffer_soft[0, :].copy()  # Start decoding using the first transmission as the base energy estimate

            if tx_num >= 1:  # Check whether the second transmission has been received
                combined_energy += rx_buffer_soft[1, :] * np.roll(rx_buffer_soft[0, :], 1)  # Combine redundancy version 1 with shifted information from version 0

            if tx_num >= 2:  # Check whether the third transmission has been received
                combined_energy += rx_buffer_soft[2, :] * np.roll(rx_buffer_soft[0, :], 3)  # Combine redundancy version 2 with shifted information from version 0

            if tx_num >= 3:  # Check whether the fourth transmission has been received
                combined_energy += rx_buffer_soft[3, :] * np.roll(rx_buffer_soft[0, :], 5)  # Combine redundancy version 3 with shifted information from version 0

            # 7. HARD DECISION
            x_hat_bits = (combined_energy < 0).astype(np.uint8)  # negative energy becomes bit 1, otherwise bit 0

            # 8. CRC CHECK 
            passed, u_hat = crc_check(x_hat_bits)  # Check whether the decoded bits pass the CRC test
            last_u_hat = u_hat  # Save the latest decoded data bits

            if passed:  # Check whether the CRC test passed
                if tx_num == 0:  # Check whether the block passed after the first transmission
                    block_log.append("Transmission 1: CRC check: PASS")  # Log that the first transmission passed CRC
                    block_log.append("ACK: decoded correctly after 1 transmission.")  # Log that the receiver sends ACK after one transmission
                    block_log.append("Final decision: ACK")  # Log the final decision as ACK
                else:  # Handle the case where the block passed after retransmissions
                    block_log.append("CRC check: PASS")  # Log that the CRC check passed
                    block_log.append(f"ACK: decoded correctly after {tx_num + 1} transmissions.")  # Log how many transmissions were needed
                    block_log.append("Final decision: ACK")  # Log the final decision as ACK

                received_payload_bits.extend(u_hat)  # Add the decoded payload bits to the final received bit list
                ack_blocks += 1  # Increase the successful block counter
                block_passed = True  # Mark this block as successfully decoded
                break  # Stop retransmitting this block because it was decoded successfully

            else:  # Handle the case where the CRC test failed
                block_log.append(f"Transmission {tx_num + 1}: CRC check: FAIL")  # Log that this transmission attempt failed CRC
                block_log.append("NACK: retransmission requested.")  # Log that the receiver requests another redundancy version

        # 9. END OF TRANSMISSION ATTEMPTS
        if not block_passed:  # Check whether the block failed after all transmission attempts
            block_log.append(f"CRC check: FAIL after {M} transmissions.")  # Log that all transmissions failed CRC
            block_log.append("Final decision: NACK")  # Log the final decision as NACK

            received_payload_bits.extend(last_u_hat)  # Add the last decoded estimate even though it may contain errors
            nack_blocks += 1  # Increase the failed block counter

        if is_early_block:  # Check whether this block is one of the first blocks selected for detailed printing
            print("\n".join(block_log))  # Print the full log for this early block

        elif not block_passed and not printed_first_nack:  # Check whether this is the first failed block outside the detailed-log range
            print("\n--- FIRST NACK ENCOUNTERED ---")  # Print a special header for the first NACK encountered
            print("\n".join(block_log))  # Print the log for this first failed block
            printed_first_nack = True  # Mark that the first NACK has already been printed

    received_payload_bits = np.array(received_payload_bits, dtype=np.uint8)[:original_length]  # Convert received bits to NumPy array and remove temporary padding

    print("\n========== True IR-HARQ SUMMARY ==========")  
    print(f"Total blocks: {total_blocks}")  # Print the number of transmitted blocks
    print(f"ACK blocks: {ack_blocks}")      # Print the number of blocks successfully decoded
    print(f"NACK blocks: {nack_blocks}")    # Print the number of blocks that failed after all attempts

    return received_payload_bits  # Return the final received bitstream with original length

# ============================================================ 
# Main Execution Block  
# ============================================================  

if __name__ == "__main__":  
    image_path = r"F:\ECE 4 2nd semester\Signal Processing\image.jpg"  # Store the path of the input image file

    if os.path.exists(image_path):  # Check whether the image file exists at the given path
        img = Image.open(image_path).convert("YCbCr")  # Open the image and convert it to YCbCr color format used in JPEG processing
        original_w, original_h = img.size  # Get the original image width and height

        new_w, new_h = original_w - (original_w % 8), original_h - (original_h % 8)  # Adjust dimensions so both are divisible by 8
        img_cropped = img.crop((0, 0, new_w, new_h))  # Crop the image to the adjusted width and height

        print(f"Original image size: {original_w}x{original_h}")  # Print the original image dimensions
        print(f"Cropped image size: {new_w}x{new_h}")  # Print the cropped image dimensions

        print("\n1. Encoding color image to JPEG-style bitstream...")  # Print the start of the image encoding step
        image_bits = encode_color_image_to_bits(img_cropped)  # Convert the cropped image into a JPEG-style binary bitstream
        print(f"Generated image bitstream: {len(image_bits):,} bits")  # Print the number of generated bits

        print("\n2. Sending image bitstream through True IR-HARQ...")  # Print the start of the HARQ transmission step

        received_bits = convolutional_ir_harq(  
            image_bits,  # Pass the original encoded image bits as input
            k=1024,  # Use 1024 payload bits per block
            M=4,  # Allow up to 4 transmissions per block
            ebn0_db=6.0,  # Use an Eb/N0 value of 6 dB
            seed=7,  # Use random seed 7 for repeatable noise generation
            verbose_blocks=10  # Print detailed logs for the first 10 blocks
        )  # End of the IR-HARQ function call

        bit_errors = np.sum(image_bits != received_bits)  # Count how many received bits differ from the original bits
        ber = bit_errors / len(image_bits)  # Calculate the bit error rate

        print(f"\nBit errors after True IR-HARQ: {bit_errors}")  # Print the total number of bit errors
        print(f"BER after True IR-HARQ: {ber:.8f}")  # Print the bit error rate with 8 decimal places

        print("\n3. Decoding received bitstream back to image...")  # Print the start of the image decoding step
        reconstructed_ycbcr = decode_bits_to_color_image(received_bits, new_h, new_w)  # Decode the received bitstream back into a YCbCr image

        output_filename = "reconstructed_color_image_IR_harq.jpeg"  # Define the output image filename
        reconstructed_ycbcr.convert("RGB").save(output_filename)  # Convert the image to RGB format and save it as a JPEG file

        print(f"\nDone! Saved output as '{output_filename}'.")  
        reconstructed_ycbcr.show()  # Display the reconstructed image on the screen

    else:  
        print(f"Error: Could not find '{image_path}'.")  # Print an error message if the image file is missing