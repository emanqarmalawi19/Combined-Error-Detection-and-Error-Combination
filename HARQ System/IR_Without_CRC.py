import numpy as np
from PIL import Image
import os

# Import JPEG encoding and decoding functions
from jpeg_colorful import encode_color_image_to_bits, decode_bits_to_color_image

# ============================================================
# Channel Functions
# ============================================================
def bpsk_mod(bits):  
    return 1.0 - 2.0 * bits.astype(np.float64)  # Math: bit 0 -> +1.0, bit 1 -> -1.0 (Digital to Analog)

def awgn_channel(symbols, ebn0_db, code_rate=1.0, rng=None):  
    if rng is None:                    
        rng = np.random.default_rng()  # Initialize random number generator for noise
    ebn0_linear = 10 ** (ebn0_db / 10.0)  # Convert dB (logarithmic) to linear scale for math
    sigma2 = 1.0 / (2.0 * code_rate * ebn0_linear)  # Calculate noise variance based on signal energy
    noise = rng.normal(0.0, np.sqrt(sigma2), size=symbols.shape)  # Generate Gaussian random noise
    return symbols + noise  # Add noise to the perfect signal to simulate wireless air travel

# ============================================================
# CRC-less IR-HARQ (Euclidean Distance / Thresholding Approach)
# ============================================================

def crcless_ir_harq(input_bits, k=1024, M=4, ebn0_db=6.0, threshold=4.0, seed=None, verbose_blocks=10):
    original_length = len(input_bits)  # Remember exact length to remove padding later
    rng = np.random.default_rng(seed)  # Seed controls randomness (None = different every time)

    pad_len = (-original_length) % k  # Calculate how many 0s we need to make bits exactly fit 1024-bit blocks
    if pad_len > 0:  
        input_bits = np.concatenate([input_bits, np.zeros(pad_len, dtype=np.uint8)])  # Add the zeros to the end

    received_payload_bits = []  # Empty list to store our final cleaned data
    total_blocks = len(input_bits) // k  # Total number of 1024-bit chunks to process
    ack_blocks = 0   # Counter for successes
    nack_blocks = 0  # Counter for total failures

    print(f"Total image bits entering CRC-less IR-HARQ: {original_length:,}")
    print(f"Block payload size k = {k}")
    print(f"Total blocks = {total_blocks}")
    print(f"Eb/N0 = {ebn0_db} dB")
    print(f"Max transmissions M = {M}")
    print(f"Reliability Threshold = {threshold}")

    printed_first_nack = False  

    for block_index in range(total_blocks):  
        u = input_bits[block_index * k:(block_index + 1) * k]  # Extract the exact 1024 bits for this block
        
        # MOTHER CODE GENERATION (Creating Math Equations from Data)
        rv0 = u  # RV0: The raw data itself (First transmission)
        rv1 = np.bitwise_xor(u, np.roll(u, 1))  # RV1: Shift data right by 1, XOR with original (Mathematical Link 1)
        rv2 = np.bitwise_xor(u, np.roll(u, 3))  # RV2: Shift data right by 3, XOR with original (Mathematical Link 2)
        rv3 = np.bitwise_xor(u, np.roll(u, 5))  # RV3: Shift data right by 5, XOR with original (Mathematical Link 3)
        
        mother_code_rvs = [rv0, rv1, rv2, rv3]  # Group them all up. Transmitter holds this in memory.
        rv_size = len(u)  # Size is 1024
        
        rx_buffer_soft = np.zeros((M, rv_size), dtype=np.float64)  # Receiver memory: 4 rows of 1024 empty float slots
        block_passed = False  
        last_c1_bits = None  # Saves the best guess just in case we fail all 4 attempts

        is_early_block = block_index < verbose_blocks
        
        block_log = []
        block_log.append(f"\n========== Block {block_index + 1}/{total_blocks} ==========")

        for tx_num in range(M):  
            tx_bits = mother_code_rvs[tx_num]  # Grab ONLY the specific slice we are sending right now
            tx_symbols = bpsk_mod(tx_bits)  # Convert 0s and 1s to +1.0 and -1.0 volts
            
            rx_symbols = awgn_channel(tx_symbols, ebn0_db, code_rate=1.0, rng=rng)  # Send through noisy air
            rx_buffer_soft[tx_num, :] = rx_symbols  # Store the received noisy float values into receiver memory row

            # 1. Soft Combining (Merge the broken pieces together to cancel noise)
            combined_energy = rx_buffer_soft[0, :].copy()  # Start with RV0 energy as base
            if tx_num >= 1:
                combined_energy += rx_buffer_soft[1, :] * np.roll(rx_buffer_soft[0, :], 1)  # Multiply RV1 parity by shifted RV0 to extract clean data
            if tx_num >= 2:
                combined_energy += rx_buffer_soft[2, :] * np.roll(rx_buffer_soft[0, :], 3)  # Use RV2 parity for extra extraction
            if tx_num >= 3:
                combined_energy += rx_buffer_soft[3, :] * np.roll(rx_buffer_soft[0, :], 5)  # Use RV3 parity (last resort)

            # 2. Find Most Likely Codeword (c1)
            c1_bits = (combined_energy < 0).astype(np.uint8)  # Hard decision: Negative energy means it's a '1', Positive means '0'
            
            # 3. Find Next Most Likely Codeword (c2)
            weakest_bit_idx = np.argmin(np.abs(combined_energy))  # Find the bit whose energy is closest to exactly 0.0 (maximum uncertainty)
            c2_bits = c1_bits.copy()  # Copy the best guess
            c2_bits[weakest_bit_idx] ^= 1  # Flip that one uncertain bit (0 becomes 1, 1 becomes 0) to make our second guess

            # 4. Re-encode c1 and c2 (Simulate what they SHOULD look like without noise to test them)
            c1_rvs = [
                c1_bits,
                np.bitwise_xor(c1_bits, np.roll(c1_bits, 1)),
                np.bitwise_xor(c1_bits, np.roll(c1_bits, 3)),
                np.bitwise_xor(c1_bits, np.roll(c1_bits, 5))
            ]
            
            c2_rvs = [
                c2_bits,
                np.bitwise_xor(c2_bits, np.roll(c2_bits, 1)),
                np.bitwise_xor(c2_bits, np.roll(c2_bits, 3)),
                np.bitwise_xor(c2_bits, np.roll(c2_bits, 5))
            ]

            # 5. Calculate Squared Euclidean Distances (How physically close are our guesses to the actual received noise?)
            dE2_c1 = 0.0
            dE2_c2 = 0.0
            for i in range(tx_num + 1):  # Only compare against the transmissions we have actually received so far
                y_i = rx_buffer_soft[i, :]  # Grab actual noisy signal from memory
                dE2_c1 += np.sum((y_i - bpsk_mod(c1_rvs[i]))**2)  # Distance formula: sum of (Actual - Perfect Guess 1)^2
                dE2_c2 += np.sum((y_i - bpsk_mod(c2_rvs[i]))**2)  # Distance formula: sum of (Actual - Perfect Guess 2)^2

            # 6. Calculate Delta (The "Confidence Gap")
            delta = dE2_c2 - dE2_c1  # High delta = c1 is much better. Low delta = they are too similar to trust.

            last_c1_bits = c1_bits  # Save c1 in case the loop finishes and we still failed

            # 7. Thresholding Decision (Replaces the traditional CRC check)
            block_log.append(f"Transmission {tx_num + 1}:")
            block_log.append(f"  -> Evaluated c1 (Most Likely) vs c2 (Next Most Likely)")
            block_log.append(f"  -> dE2(y, c1) = {dE2_c1:.2f}")
            block_log.append(f"  -> dE2(y, c2) = {dE2_c2:.2f}")
            block_log.append(f"  -> Delta = {delta:.2f} | Threshold = {threshold:.2f}")

            if delta >= threshold:  # If Delta is >= threshold, confidence is strong enough to accept the data
                block_log.append(f"  -> Result: Delta >= Threshold. Reliability is HIGH.")
                block_log.append(f"  -> ACK: decoded correctly after {tx_num + 1} transmissions.")
                block_log.append(f"Final decision: ACK")
                
                received_payload_bits.extend(c1_bits)  # Add the good bits to our final output list
                ack_blocks += 1      
                block_passed = True  
                break  # SUCCESS! Break out of the 'tx_num' loop so we don't ask for any more parity bits
            else:  # If Delta is < threshold, the noise was too heavy. We don't trust c1 yet.
                block_log.append(f"  -> Result: Delta < Threshold. Reliability is LOW.")
                block_log.append("  -> NACK: retransmission requested.") # Loop continues to next tx_num to get more parity

        if not block_passed:  # If we went through all attempts (M) and STILL failed the threshold
            block_log.append("Final decision: NACK")
            received_payload_bits.extend(last_c1_bits)  # Force accept the best guess anyway so the image pixels don't misalign
            nack_blocks += 1  

        if is_early_block:
            print("\n".join(block_log))
        elif not block_passed and not printed_first_nack:
            print("\n--- FIRST NACK ENCOUNTERED ---")
            print("\n".join(block_log))
            printed_first_nack = True

    received_payload_bits = np.array(received_payload_bits, dtype=np.uint8)[:original_length]  # Chop off the zeros we padded at the start
    
    print("\n========== CRC-less IR-HARQ SUMMARY ==========")
    print(f"Total blocks: {total_blocks}")
    print(f"ACK blocks: {ack_blocks}")
    print(f"NACK blocks: {nack_blocks}")

    return received_payload_bits  
    
# ============================================================
# Main Execution Block
# ============================================================

if __name__ == "__main__": 
    image_path = r"F:\ECE 4 2nd semester\Signal Processing\image.jpg"
    
    if os.path.exists(image_path):  
        img = Image.open(image_path).convert("YCbCr")  # Convert to standard JPEG color space
        original_w, original_h = img.size
        new_w, new_h = original_w - (original_w % 8), original_h - (original_h % 8) # Make sure dimensions are divisible by 8 for compression
        img_cropped = img.crop((0, 0, new_w, new_h))  
        
        print("\n1. Encoding color image to JPEG-style bitstream...")
        image_bits = encode_color_image_to_bits(img_cropped)  # Compress to array of 0s and 1s
        
        print("\n2. Sending image bitstream through CRC-less IR-HARQ...")
        received_bits = crcless_ir_harq(
            image_bits, 
            k=1024, 
            M=4, 
            ebn0_db=6.0, 
            threshold=4.0,  
            seed=7, 
            verbose_blocks=10
        )

        bit_errors = np.sum(image_bits != received_bits) # Count how many bits flipped
        ber = bit_errors / len(image_bits) # Bit Error Rate

        print(f"\nBit errors after CRC-less IR-HARQ: {bit_errors}")
        print(f"BER after CRC-less IR-HARQ: {ber:.8f}")              

        print("\n3. Decoding received bitstream back to image...")
        reconstructed_ycbcr = decode_bits_to_color_image(received_bits, new_h, new_w)  # Rebuild image from bits
        
        output_filename = "reconstructed_color_image_crcless_ir.jpeg"
        reconstructed_ycbcr.convert("RGB").save(output_filename) # Save to disk
        
        print(f"\nDone! Saved output as '{output_filename}'.")
        reconstructed_ycbcr.show() # Open the image viewer
    else:
        print(f"Error: Could not find '{image_path}'.")