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
    original_length = len(input_bits)  
    rng = np.random.default_rng(seed)  

    # Calculate if the image bits do not divide perfectly into blocks of size 'k'
    pad_len = (-original_length) % k  
    if pad_len > 0:  
        # Add temporary zeros to the end to make it divide perfectly
        input_bits = np.concatenate([input_bits, np.zeros(pad_len, dtype=np.uint8)])  

    received_payload_bits = []           
    total_blocks = len(input_bits) // k  
    ack_blocks = 0   
    nack_blocks = 0  

    # Print simulation configuration headers
    print(f"Total image bits entering True IR-HARQ: {original_length:,}")
    print(f"Block payload size k = {k}")
    print(f"Total blocks = {total_blocks}")
    print(f"Eb/N0 = {ebn0_db} dB")
    print(f"Max transmissions M = {M}")

    printed_first_nack = False  

    # Process the entire image, one block (e.g., 1024 bits) at a time
    for block_index in range(total_blocks):  
        
        # 1. Extract the specific 1024 bits for the current block
        u = input_bits[block_index * k:(block_index + 1) * k]  
        
        # 2. Append the 8-bit CRC checksum to this block
        u_crc = crc_encode(u)       
        
        # 3. MOTHER CODE GENERATION (Creating Parity using Convolutional logic)
        rv0 = u_crc # RV0 is just the pure systematic data (and CRC)
        rv1 = np.bitwise_xor(u_crc, np.roll(u_crc, 1))
        rv2 = np.bitwise_xor(u_crc, np.roll(u_crc, 3))
        rv3 = np.bitwise_xor(u_crc, np.roll(u_crc, 5)) 
        
        mother_code_rvs = [rv0, rv1, rv2, rv3]
        rv_size = len(u_crc)
        
        # 4. RECEIVER INITIALIZATION
        rx_buffer_soft = np.zeros((M, rv_size), dtype=np.float64)        
        block_passed = False  
        last_u_hat = None     

        is_early_block = block_index < verbose_blocks
        
        block_log = []
        block_log.append(f"\n========== Block {block_index + 1}/{total_blocks} ==========")

        # 5. TRANSMISSION LOOP (Up to M attempts allowed per block)
        for tx_num in range(M):  
            
            tx_bits = mother_code_rvs[tx_num]
            tx_symbols = bpsk_mod(tx_bits)  
            rx_symbols = awgn_channel(tx_symbols, ebn0_db, code_rate=1.0, rng=rng)  
            rx_buffer_soft[tx_num, :] = rx_symbols

            # 6. DECODING & SOFT COMBINING
            combined_energy = rx_buffer_soft[0, :].copy()
            
            if tx_num >= 1:
                combined_energy += rx_buffer_soft[1, :] * np.roll(rx_buffer_soft[0, :], 1)
            if tx_num >= 2:
                combined_energy += rx_buffer_soft[2, :] * np.roll(rx_buffer_soft[0, :], 3)
            if tx_num >= 3:
                combined_energy += rx_buffer_soft[3, :] * np.roll(rx_buffer_soft[0, :], 5)

            # 7. HARD DECISION
            x_hat_bits = (combined_energy < 0).astype(np.uint8)
            
            # 8. CRC CHECK
            passed, u_hat = crc_check(x_hat_bits)  
            last_u_hat = u_hat                

            if passed:  
                if tx_num == 0:
                    block_log.append("Transmission 1: CRC check: PASS")
                    block_log.append("ACK: decoded correctly after 1 transmission.")
                    block_log.append("Final decision: ACK")
                else:
                    block_log.append("CRC check: PASS")
                    block_log.append(f"ACK: decoded correctly after {tx_num + 1} transmissions.")
                    block_log.append("Final decision: ACK")
                
                received_payload_bits.extend(u_hat)  
                ack_blocks += 1      
                block_passed = True  
                break  
            else:
                block_log.append(f"Transmission {tx_num + 1}: CRC check: FAIL")
                block_log.append("NACK: retransmission requested.")

        # 9. END OF TRANSMISSION ATTEMPTS
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
            printed_first_nack = True

    # Remove the temporary padding zeros we added at the beginning
    received_payload_bits = np.array(received_payload_bits, dtype=np.uint8)[:original_length]          
    
    # Match the summary block format exactly
    print("\n========== True IR-HARQ SUMMARY ==========")
    print(f"Total blocks: {total_blocks}")
    print(f"ACK blocks: {ack_blocks}")
    print(f"NACK blocks: {nack_blocks}")

    return received_payload_bits

# ============================================================
# Main Execution Block
# ============================================================
if __name__ == "__main__": 
    # Define where the image is stored on your computer
    image_path = r"F:\ECE 4 2nd semester\Signal Processing\image.jpg"
    
    if os.path.exists(image_path):  
        # Open the image and convert it to YCbCr color space (standard for JPEG)
        img = Image.open(image_path).convert("YCbCr")  
        original_w, original_h = img.size
        
        # Crop the image slightly so its dimensions are perfectly divisible by 8
        new_w, new_h = original_w - (original_w % 8), original_h - (original_h % 8)
        img_cropped = img.crop((0, 0, new_w, new_h))  
        
        print(f"Original image size: {original_w}x{original_h}")
        print(f"Cropped image size: {new_w}x{new_h}")

        print("\n1. Encoding color image to JPEG-style bitstream...")
        # Compress and convert the image pixels into an array of bits
        image_bits = encode_color_image_to_bits(img_cropped)  
        print(f"Generated image bitstream: {len(image_bits):,} bits")
        
        print("\n2. Sending image bitstream through True IR-HARQ...")
        
        # Pass the bitstream through our wireless simulation
        received_bits = convolutional_ir_harq(
            image_bits, 
            k=1024, 
            M=4, 
            ebn0_db=6.0, 
            seed=7, 
            verbose_blocks=10
        )

        # Count how many bits flip compared to the original
        bit_errors = np.sum(image_bits != received_bits)
        ber = bit_errors / len(image_bits)

        print(f"\nBit errors after True IR-HARQ: {bit_errors}")
        print(f"BER after True IR-HARQ: {ber:.8f}")              

        print("\n3. Decoding received bitstream back to image...")
        # Convert the received (and possibly slightly corrupted) bits back into image pixels
        reconstructed_ycbcr = decode_bits_to_color_image(received_bits, new_h, new_w)  
        
        # Convert back to normal RGB colors and save to disk
        output_filename = "reconstructed_color_image_IR_harq.jpeg"
        reconstructed_ycbcr.convert("RGB").save(output_filename)
        
        print(f"\nDone! Saved output as '{output_filename}'.")
        # Display the final image on screen
        reconstructed_ycbcr.show()
    else:
        # Failsafe if the image path is incorrect
        print(f"Error: Could not find '{image_path}'.")