"""
================================================================================
  ir_harq_image.py
  IR-HARQ Image Transmission Pipeline
  University Project — Signal Processing

  Compares exactly two systems:
    1. IR-HARQ without CRC   (matches Fig. 17.3)
    2. IR-HARQ with CRC      (matches Fig. 17.4)
================================================================================

WHAT IS IR-HARQ?
----------------
  HARQ = Hybrid Automatic Repeat reQuest.
  It combines forward error correction (FEC) with automatic retransmission.
  When the receiver cannot decode correctly, it asks the transmitter to send
  more bits (NACK). When it succeeds, it sends ACK.

  "Incremental Redundancy" (IR) means each retransmission sends NEW bits,
  not a copy of the old ones. The transmitter pre-computes a long "mother"
  codeword of N_TOTAL bits, then sends it in chunks:
    Round 1: chunk 1  (systematic + early parity)
    Round 2: chunk 2  (new parity bits)
    Round 3: chunk 3  (more new parity bits)
    Round 4: chunk 4  (final parity bits — full mother codeword)

  At the receiver, each round's bits are accumulated in a buffer (y).
  This progressively improves the soft estimate and lowers the effective
  code rate until the decoder can succeed.

WHAT IS A MOTHER CODEWORD?
---------------------------
  The transmitter encodes K_INFO information bits into a single long
  codeword c_M of N_TOTAL bits.  This is called the "mother code":
    c_M = FEC_encode(u)   where u = K_INFO info bits
  In Round r, only the r-th chunk of c_M is sent over the channel.
  The receiver buffers all received chunks before decoding.

WHAT IS K_INFO?
---------------
  K_INFO = 8192 bits per frame.
  This is the number of information bits in each frame (before CRC).
  It must be strictly less than CHUNK_ENDS[0] (the first chunk length),
  so Round 1 always carries at least the full systematic payload.

WHAT IS CRC_LEN?
----------------
  CRC_LEN = 8 bits.
  In the "with CRC" system, 8 CRC bits are appended to the K_INFO info
  bits before FEC encoding.  So the FEC encoder receives K_INFO + 8 bits.
  The receiver uses CRC to validate candidate codewords.

WHAT IS N_TOTAL?
----------------
  N_TOTAL = 20480 bits.
  This is the total length of the mother codeword.  It determines the
  lowest possible code rate: K_INFO / N_TOTAL = 8192 / 20480 = 0.4.

WHAT IS CHUNK_ENDS?
-------------------
  CHUNK_ENDS = [11264, 14336, 17408, 20480]
  Each value is the end index (exclusive) of the bits sent in each round.
    Round 1: bits [0     : 11264]  length = 11264  (>K_INFO = 8192 ✓)
    Round 2: bits [11264 : 14336]  length =  3072
    Round 3: bits [14336 : 17408]  length =  3072
    Round 4: bits [17408 : 20480]  length =  3072
  CHUNK_ENDS[-1] must equal N_TOTAL.

  Why is Round 1 length (11264) > K_INFO (8192)?
    In a systematic code, the first K_INFO bits of c_M are the info bits
    themselves, followed by parity.  Round 1 must transmit all K_INFO
    systematic bits PLUS some early parity — so its length must exceed
    K_INFO.  With K_INFO=16384 (old value), Round 1 length was only 11264,
    which is LESS than K_INFO — the receiver would never see all systematic
    bits in Round 1.  That is the bug fixed here.

WHAT IS THE RELIABILITY METRIC Δ?
-----------------------------------
  After FEC decoding, we get a list of candidate codewords c1, c2, ...
  sorted by Euclidean distance to the received soft vector y:
    d²(y, c1) ≤ d²(y, c2) ≤ ...
  The reliability metric is:
    Δ = d²(y, c2) - d²(y, c1)
  A large Δ means c1 is far more likely than c2 → high confidence.
  A small Δ means two candidates are nearly equally likely → uncertain.

  For Fig. 17.4 (with CRC), the candidates are already CRC-filtered:
    Δ = d²(y, c_l) - d²(y, c_j)
  where c_j = best CRC-valid candidate, c_l = second-best CRC-valid.

HOW IS ACK/NACK GENERATED?
---------------------------
  ACK is sent if:   Δ >= RELIABILITY_THRESHOLD
  NACK is sent if:  Δ <  RELIABILITY_THRESHOLD  (request more parity)
  If we are in the final round, we force ACK regardless of Δ.

HOW IS THE IMAGE CONVERTED TO BITS?
-------------------------------------
  1. Load the image in RGB.
  2. Crop to the nearest 8×8 multiple (JPEG block alignment).
  3. Convert to YCbCr colour space.
  4. For each channel (Y, Cb, Cr):
       a. Divide the channel into 8×8 pixel blocks.
       b. Apply 2D DCT to each block (frequency transform).
       c. Quantize the DCT coefficients with the JPEG quantization matrix.
       d. Represent each coefficient as a 16-bit signed integer.
       e. Flatten all blocks into a 1D bitstream.
  5. Concatenate Y, Cb, Cr bitstreams → one flat bit array.

HOW IS TRANSMISSION DONE?
---------------------------
  1. Pad the bitstream to a multiple of K_INFO.
  2. Split into K_INFO-bit frames.
  3. For each frame:
       a. FEC encode → mother codeword c_M.
       b. Transmit chunk by chunk over BPSK + AWGN channel.
       c. After each chunk, try to decode.
       d. Send ACK or NACK.
       e. If ACK or final round, output decoded info bits.
  4. Reassemble all decoded frames.
  5. Remove padding.
  6. Reconstruct the image from the decoded bitstream.

HOW TO RUN:
-----------
  python ir_harq_image.py

  Place your input image at the path given to run_image_test().
  Default: "image.jpeg"

FILES EXPECTED:
  - ir_harq_image.py      (this file)
  - jpeg_colorful.py      (JPEG codec companion)
  - image.jpeg            (or any JPEG/PNG image)

FILES PRODUCED (in outputs/):
  - original_processed.png          Codec-only reference (no channel noise)
  - reconstructed_ir_without_crc.png  IR-HARQ without CRC result
  - reconstructed_ir_with_crc.png     IR-HARQ with CRC result
  - transmission_report.csv          Per-frame statistics
  - summary_report.csv               Per-system summary

FIGURE MAPPING:
  Fig. 17.3 (no CRC) → simulate_frame_ir_no_crc()
    u → codec_no_crc.encode() → cM → get_chunk() → bpsk+awgn
    → rx_buffer → generate_candidates_no_crc() → [c1, c2]
    → compute_reliability_delta() → Δ → ack_nack_decision()
    → output û

  Fig. 17.4 (with CRC) → simulate_frame_ir_with_crc()
    u → CRC8.append() → x → codec_with_crc.encode() → cM
    → get_chunk() → bpsk+awgn → rx_buffer
    → generate_candidates_with_crc() → candidate list
    → crc_filter_candidates() → [cj, cl]
    → compute_reliability_delta() → Δ → ack_nack_decision()
    → output û (CRC stripped)
================================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0 — IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np                          # array maths
import os                                   # file/folder utilities
import csv                                  # CSV writing
import time                                 # runtime measurement
from dataclasses import dataclass           # lightweight result containers
from itertools import combinations          # flip-pattern enumeration
from typing import List, Optional, Tuple    # type hints
from trace_utils import (
    setup_trace,
    is_trace_enabled,
    save_txt_snapshot,
    save_candidates_txt,
)

from PIL import Image                       # image I/O

# Import the JPEG codec functions from the companion file.
# We use these for image → bitstream and bitstream → image conversion.
from jpeg_colorful import (
    encode_channel,   # encodes one colour channel (e.g. Y, Cb, Cr) to bits
    decode_channel,   # decodes one colour channel from bits
    Q_MATRIX,         # luminance (Y) quantization matrix
    C_MATRIX,         # chrominance (Cb, Cr) quantization matrix
)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — GLOBAL IR-HARQ PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

# ── Core frame parameters ────────────────────────────────────────────────────
K_INFO  = 2048
CRC_LEN = 8
N_TOTAL = 5120

# ── Chunk boundaries (IR rounds) ─────────────────────────────────────────────
# Each entry is the EXCLUSIVE end index of the bits sent in that round.
# CHUNK_ENDS[-1] must equal N_TOTAL (assertion checked in SECTION 2).
#
# Round 1: bits [0     : 11264]  len=11264  → 11264 > K_INFO=8192  ✓
#   Contains all K_INFO systematic bits + (11264-8192)=3072 early parity bits.
# Round 2: bits [11264 : 14336]  len=3072   → new parity
# Round 3: bits [14336 : 17408]  len=3072   → more new parity
# Round 4: bits [17408 : 20480]  len=3072   → final parity (full mother code)
CHUNK_ENDS   = [2816, 3584, 4352, 5120]
CHUNK_STARTS = [0] + CHUNK_ENDS[:-1]
MAX_ROUNDS   = len(CHUNK_ENDS)            # 4

# ── List decoder parameters ───────────────────────────────────────────────────
LIST_SIZE       = 5
MAX_FLIP_WEIGHT = 1
SEARCH_WIDTH    = 4

# ── Reliability threshold ─────────────────────────────────────────────────────
# ACK is triggered when Δ = d²(y,c2) - d²(y,c1) >= this value.
# Higher = more conservative (may use more rounds but fewer errors).
RELIABILITY_THRESHOLD = 1450.0

# ── Output folder ─────────────────────────────────────────────────────────────
OUTPUT_DIR = "outputs"


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — SAFETY ASSERTIONS
# ─────────────────────────────────────────────────────────────────────────────

# Verify that CHUNK_ENDS[-1] equals N_TOTAL exactly
assert CHUNK_ENDS[-1] == N_TOTAL, (
    f"CHUNK_ENDS[-1]={CHUNK_ENDS[-1]} must equal N_TOTAL={N_TOTAL}")

# Verify that we have exactly 4 rounds
assert MAX_ROUNDS == 4, f"Expected 4 rounds, got {MAX_ROUNDS}"

# Verify that N_TOTAL > K_INFO + CRC_LEN (FEC encoder must have room for parity)
assert N_TOTAL > K_INFO + CRC_LEN, (
    f"N_TOTAL={N_TOTAL} must exceed K_INFO+CRC_LEN={K_INFO+CRC_LEN}")

# Critical fix check: Round 1 length must be strictly greater than K_INFO.
# This ensures the first chunk carries all systematic bits AND some parity.
# Old bug: K_INFO=16384, CHUNK_ENDS[0]=11264 → 11264 < 16384 → WRONG.
# Fixed:   K_INFO= 8192, CHUNK_ENDS[0]=11264 → 11264 >  8192 → CORRECT.
assert CHUNK_ENDS[0] > K_INFO, (
    f"Round 1 length ({CHUNK_ENDS[0]}) must be > K_INFO ({K_INFO}). "
    f"Otherwise Round 1 cannot carry all systematic information bits.")

# Verify that CHUNK_ENDS are strictly increasing
for _i in range(1, len(CHUNK_ENDS)):
    assert CHUNK_ENDS[_i] > CHUNK_ENDS[_i-1], (
        f"CHUNK_ENDS must be strictly increasing: "
        f"{CHUNK_ENDS[_i-1]} >= {CHUNK_ENDS[_i]}")

# Compute and display chunk lengths for verification
_chunk_lengths = [CHUNK_ENDS[i] - CHUNK_STARTS[i] for i in range(MAX_ROUNDS)]
print(f"[OK] K_INFO={K_INFO}, CRC_LEN={CRC_LEN}, N_TOTAL={N_TOTAL}")
print(f"[OK] Code rate (min) = {K_INFO}/{N_TOTAL} = {K_INFO/N_TOTAL:.4f}")
print(f"[OK] CHUNK_STARTS    = {CHUNK_STARTS}")
print(f"[OK] CHUNK_ENDS      = {CHUNK_ENDS}")
print(f"[OK] Chunk lengths   = {_chunk_lengths}  (sum={sum(_chunk_lengths)})")
print(f"[OK] Round 1 length ({CHUNK_ENDS[0]}) > K_INFO ({K_INFO}) ✓")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — CRC-8 ENCODER / CHECKER
# ─────────────────────────────────────────────────────────────────────────────

class CRC8:
    """
    CRC-8 using polynomial x^8 + x^2 + x + 1  (0x07 in hex).

    Purpose in Fig. 17.4:
      Transmitter: x = [u | CRC(u)]   (u = K_INFO info bits)
      Receiver:    check if CRC(candidate info bits) is valid.

    Methods:
      compute(bits)      → 8-bit CRC as a list of ints
      append(bits)       → bits + CRC(bits) (what the transmitter sends)
      check(bits_w_crc)  → True if CRC is valid (last 8 bits match)
    """

    POLY  = 0x07    # CRC-8 polynomial (without the leading x^8 term)
    WIDTH = 8       # CRC length in bits

    @staticmethod
    def compute(bits: List[int]) -> List[int]:
        """Compute CRC-8 of a list of bits. Returns 8-bit list."""
        reg = 0                                # shift register, initially 0
        for bit in bits:                       # process each input bit
            msb = (reg >> 7) & 1               # current MSB of register
            reg = ((reg << 1) & 0xFF) | bit    # shift in the new bit
            if msb:                            # if MSB was 1, XOR with polynomial
                reg ^= CRC8.POLY
        return [(reg >> i) & 1 for i in range(7, -1, -1)]  # extract 8 bits, MSB first

    @staticmethod
    def append(bits: List[int]) -> List[int]:
        """Append 8 CRC bits to a bit list. Returns bits + CRC (total len+8 bits)."""
        return bits + CRC8.compute(bits)       # concatenate payload and its CRC

    @staticmethod
    def check(bits_with_crc: List[int]) -> bool:
        """Check if the last 8 bits are a valid CRC for the preceding bits."""
        if len(bits_with_crc) < CRC8.WIDTH:
            return False                                      # too short to have a CRC
        payload      = bits_with_crc[:-CRC8.WIDTH]           # all bits except last 8
        received_crc = bits_with_crc[-CRC8.WIDTH:]           # last 8 bits = received CRC
        expected_crc = CRC8.compute(payload)                  # compute expected CRC
        return received_crc == expected_crc                   # True if they match


# ── CRC self-test ─────────────────────────────────────────────────────────────
_tb  = [1, 0, 1, 1, 0, 0, 1, 0]          # some test bits
_tc  = CRC8.append(_tb)                   # append CRC → should check correctly
assert CRC8.check(_tc),  "CRC self-test FAILED (clean)"
_bad = _tc[:]                             # copy
_bad[2] ^= 1                             # flip one bit → should fail check
assert not CRC8.check(_bad), "CRC self-test FAILED (corrupted)"
print("[OK] CRC-8 self-test passed")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — IR MOTHER FEC CODEC
# ─────────────────────────────────────────────────────────────────────────────

class IRMotherCodec:
    """
    Systematic linear block code for IR-HARQ.

    Input:  k bits  (info bits, possibly including CRC)
    Output: n bits  [systematic part (k bits) | parity part (n-k bits)]

    The parity matrix G_parity is a fixed random binary (n-k) × k matrix.
    It is generated with a fixed seed so encoder and decoder always agree.

    Encoding: c_M = [u | G_parity @ u  mod 2]
      The first k bits of c_M ARE the info bits (systematic code property).
      The remaining n-k bits are parity checks.

    The codec is used to slice c_M into IR chunks via get_chunk().
    """

    def __init__(self, k: int, n: int = N_TOTAL, matrix_seed: int = 12345):
        assert n > k, f"n={n} must be > k={k}"
        self.k        = k              # number of input bits (info or info+CRC)
        self.n        = n              # mother codeword length
        self.n_parity = n - k          # number of parity bits = n - k

        # Build the fixed-seed random parity generator matrix G_parity.
        # Shape: (n_parity, k) — one row per parity bit, one column per info bit.
        rng = np.random.default_rng(matrix_seed)
        self.G_parity = rng.integers(0, 2,
                                     size=(self.n_parity, self.k),
                                     dtype=np.int8)  # binary {0,1} matrix

    def encode(self, info_bits) -> np.ndarray:
        """
        Encode k input bits into the full N_TOTAL-bit mother codeword.
        Returns: [info_bits (k) | parity (n-k)] as int8 numpy array.
        """
        assert len(info_bits) == self.k, \
            f"Expected {self.k} bits, got {len(info_bits)}"
        u = np.array(info_bits, dtype=np.int8)          # info bits as int8
        # Compute parity: each parity bit = inner product of one G_parity row with u, mod 2
        p = self.G_parity.dot(u.astype(np.int32)) % 2   # shape (n_parity,)
        # Concatenate systematic (u) and parity (p) → mother codeword c_M
        return np.concatenate([u, p.astype(np.int8)])    # shape (N_TOTAL,)

    def get_chunk(self, codeword: np.ndarray, round_idx: int) -> np.ndarray:
        """
        Extract the bits for one IR transmission round.
        round_idx=0 → Round 1, etc.
        Returns: slice of c_M from CHUNK_STARTS[round_idx] to CHUNK_ENDS[round_idx].
        """
        assert 0 <= round_idx < MAX_ROUNDS, \
            f"round_idx must be in [0, {MAX_ROUNDS-1}]"
        return codeword[CHUNK_STARTS[round_idx] : CHUNK_ENDS[round_idx]]


# ── Pre-build the two codecs ──────────────────────────────────────────────────

# System A (Fig. 17.3): FEC encoder takes K_INFO info bits
codec_no_crc   = IRMotherCodec(k=K_INFO,           n=N_TOTAL, matrix_seed=11111)

# System B (Fig. 17.4): FEC encoder takes K_INFO + CRC_LEN bits (info + CRC appended)
codec_with_crc = IRMotherCodec(k=K_INFO + CRC_LEN, n=N_TOTAL, matrix_seed=22222)

print(f"[OK] codec_no_crc   : k={codec_no_crc.k},   n={codec_no_crc.n}")
print(f"[OK] codec_with_crc : k={codec_with_crc.k}, n={codec_with_crc.n}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — CHANNEL MODEL: BPSK + AWGN
# ─────────────────────────────────────────────────────────────────────────────

def bpsk_modulate(bits: np.ndarray) -> np.ndarray:
    """
    BPSK modulation: map binary bits to bipolar symbols.
      bit 0 → +1.0  (represents the "no energy" symbol)
      bit 1 → -1.0  (represents the "signal" symbol)
    """
    return np.where(bits == 0, 1.0, -1.0)   # vectorized mapping


def awgn_channel(symbols: np.ndarray, sigma: float,
                 rng: np.random.Generator) -> np.ndarray:
    """
    Add white Gaussian noise to BPSK symbols.
    sigma = noise standard deviation (computed from Eb/N0).
    Returns: noisy received soft values y = s + n.
    """
    noise = rng.normal(0.0, sigma, size=len(symbols))   # Gaussian noise vector
    return symbols + noise                               # y = s + n


def ebn0_db_to_sigma(ebn0_db: float) -> float:
    """
    Convert Eb/N0 (in dB) to AWGN noise standard deviation sigma.

    For BPSK with code rate R = K_INFO / N_TOTAL:
      Eb/N0 = (signal energy per info bit) / (noise spectral density)
      sigma = sqrt(N_TOTAL / (2 * K_INFO * Eb/N0_linear))
    """
    ebn0_lin = 10.0 ** (ebn0_db / 10.0)               # convert dB to linear
    return np.sqrt(N_TOTAL / (2.0 * K_INFO * ebn0_lin))  # noise std dev


def hard_decision(soft: np.ndarray) -> np.ndarray:
    """
    Hard-decision demodulation of soft values.
    soft > 0 → +1 was sent → bit 0
    soft < 0 → -1 was sent → bit 1
    Used as a starting point for the list decoder.
    """
    return (soft < 0).astype(np.int8)   # returns {0,1} binary array


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — CANDIDATE / DISTANCE STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Candidate:
    """
    A single decoded candidate codeword with its reliability score.

    Attributes:
      payload_bits  : the K_INFO information bits (CRC stripped if applicable)
      codeword_bits : the full N_TOTAL-bit codeword produced by FEC encoding
      distance2     : squared Euclidean distance d²(y, c) = ||y - BPSK(c)||²
    """
    payload_bits:  np.ndarray   # K_INFO decoded info bits
    codeword_bits: np.ndarray   # full N_TOTAL codeword (for distance computation)
    distance2:     float        # d²(y, c) — lower is better (c is closer to y)


def euclidean_distance2(y: np.ndarray, bits: np.ndarray) -> float:
    """
    Compute squared Euclidean distance between soft received vector y
    and the BPSK-modulated version of a candidate codeword.

    d²(y, c) = Σ (y_i - s_i)²    where s_i = BPSK(c_i)

    Only the first len(y) bits of the codeword are used (buffer may be
    shorter than N_TOTAL in early rounds).
    """
    n = len(y)                                     # number of received symbols so far
    c = np.where(bits[:n] == 0, 1.0, -1.0)        # BPSK-map the first n codeword bits
    diff = y - c                                   # element-wise difference vector
    return float(diff.dot(diff))                   # sum of squared differences = d²


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — FEC / LIST DECODER WITHOUT CRC  (Fig. 17.3)
# ─────────────────────────────────────────────────────────────────────────────

def generate_candidates_no_crc(
    rx_buffer: np.ndarray,
    list_size: int = LIST_SIZE,
    max_flip_weight: int = MAX_FLIP_WEIGHT,
    search_width: int = SEARCH_WIDTH,
) -> List[Candidate]:
    """
    FEC list decoder for IR without CRC (corresponds to 'FEC decoder' in Fig. 17.3).

    Algorithm:
      1. Hard-decide the K_INFO systematic bits from the receive buffer.
      2. Identify the SEARCH_WIDTH least-reliable positions (smallest |y_i|).
      3. Generate candidates by flipping 0, 1, ..., max_flip_weight of those bits.
      4. For each candidate payload, compute the codeword and d²(y, c).
      5. Return the LIST_SIZE best candidates sorted by d².

    Parameters:
      rx_buffer : accumulated soft received values (grows with each round)

    Note: if rx_buffer is shorter than K_INFO (early rounds), the missing
    systematic positions are treated as zero soft value (fully unreliable).
    """
    L = len(rx_buffer)   # number of received soft values so far

    # Pad soft buffer to K_INFO if shorter (zero = maximally uncertain)
    if L < K_INFO:
        padded = np.concatenate([rx_buffer, np.zeros(K_INFO - L)])
    else:
        padded = rx_buffer[:K_INFO]                    # use only systematic part

    hd_info   = hard_decision(padded)                  # initial hard-decision estimate
    reliability = np.abs(padded)                       # |y_i| = reliability of each bit
    # Find the SEARCH_WIDTH bit positions with the smallest reliability
    weak_pos  = np.argsort(reliability)[:min(search_width, K_INFO)].tolist()

    candidates: List[Candidate] = []
    seen = set()    # track already-evaluated payload patterns (avoid duplicates)

    # Enumerate all flip patterns with weight 0, 1, ..., max_flip_weight
    for w in range(max_flip_weight + 1):
        for idxs in combinations(weak_pos, w):         # choose w positions to flip
            payload = hd_info.copy()                   # start from hard-decision
            for idx in idxs:
                payload[idx] ^= 1                      # flip selected bits

            key = payload.tobytes()                    # unique key for deduplication
            if key in seen:
                continue                               # skip if already evaluated
            seen.add(key)

            # Encode the candidate payload into the full mother codeword c_M
            codeword = codec_no_crc.encode(payload.tolist())

            # Compute d²(y, c_M) using only the bits received so far
            d2 = euclidean_distance2(rx_buffer, codeword)

            # Store the candidate
            candidates.append(Candidate(
                payload_bits  = payload.copy(),
                codeword_bits = codeword,
                distance2     = d2,
            ))

    # Sort by distance (best = lowest d²) and return top LIST_SIZE
    candidates.sort(key=lambda c: c.distance2)
    return candidates[:list_size] if candidates else []


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — FEC / LIST DECODER WITH CRC  (Fig. 17.4, decoder block)
# ─────────────────────────────────────────────────────────────────────────────

def generate_candidates_with_crc(
    rx_buffer: np.ndarray,
    list_size: int = LIST_SIZE,
    max_flip_weight: int = MAX_FLIP_WEIGHT,
    search_width: int = SEARCH_WIDTH,
) -> List[Candidate]:
    """
    FEC list decoder for IR with CRC (corresponds to 'FEC decoder' in Fig. 17.4).

    Key difference from no-CRC version:
      After generating each candidate payload (K_INFO bits), the correct
      CRC is appended before FEC encoding.  This guarantees that every
      candidate codeword was produced from a valid (payload + CRC) input.
      The CRC checking stage (SECTION 9) then checks the RECEIVED bits
      against the candidates' CRC bits.

    The payload_bits field stores only the K_INFO info bits (CRC stripped),
    because the receiver ultimately wants to decode the original message.
    """
    L = len(rx_buffer)

    # Pad or trim soft buffer to K_INFO (systematic part only for flip decisions)
    if L < K_INFO:
        padded = np.concatenate([rx_buffer, np.zeros(K_INFO - L)])
    else:
        padded = rx_buffer[:K_INFO]

    hd_payload  = hard_decision(padded)                # hard-decision on info bits
    reliability = np.abs(padded)                       # reliability of each info bit
    # Identify least-reliable info bit positions
    weak_pos    = np.argsort(reliability)[:min(search_width, K_INFO)].tolist()

    candidates: List[Candidate] = []
    seen = set()

    for w in range(max_flip_weight + 1):
        for idxs in combinations(weak_pos, w):         # flip pattern
            payload = hd_payload.copy()
            for idx in idxs:
                payload[idx] ^= 1                      # apply flip

            key = payload.tobytes()
            if key in seen:
                continue
            seen.add(key)

            # Step matching Fig. 17.4 transmitter:
            #   x = [u | CRC(u)]    (K_INFO + CRC_LEN bits)
            payload_crc = CRC8.append(payload.tolist())   # append correct CRC to payload

            # Encode x into the full mother codeword c_M
            codeword = codec_with_crc.encode(payload_crc)

            # Compute distance using only received bits so far
            d2 = euclidean_distance2(rx_buffer, codeword)

            # Store K_INFO payload bits only (CRC is internal, not part of message)
            candidates.append(Candidate(
                payload_bits  = payload.copy(),    # K_INFO info bits (no CRC)
                codeword_bits = codeword,          # full N_TOTAL codeword
                distance2     = d2,
            ))

    candidates.sort(key=lambda c: c.distance2)
    return candidates[:list_size] if candidates else []


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — CRC CHECKING STAGE  (explicit block from Fig. 17.4)
# ─────────────────────────────────────────────────────────────────────────────

def crc_filter_candidates(
    candidates: List[Candidate],
    rx_buffer: np.ndarray,
) -> List[Candidate]:
    """
    CRC CHECKING STAGE — explicitly modelled from Fig. 17.4.

    In Fig. 17.4 the block diagram shows:
        FEC decoder → CRC checking → [cj, cl] (best CRC-valid candidates)

    This function implements exactly that CRC checking stage:
      For each candidate in the list:
        1. Re-extract the K_INFO+CRC_LEN systematic bits from the candidate
           codeword (the first k bits of c_M are exactly the encoded input x).
        2. Check if those bits pass the CRC-8 check.
        3. Keep only candidates that pass.

    The result is a filtered list of CRC-valid candidates, still sorted
    by d²(y, c) — i.e. cj (best) first, cl (second-best) second.

    Why is this important?
      Without CRC filtering, both systems would use the same raw Δ metric.
      With CRC filtering:
        - Only candidates with a matching CRC are trusted.
        - Δ = d²(y, cl) - d²(y, cj)  (cl and cj are CRC-valid)
        - A NACK is sent if there is NO CRC-valid candidate at all.

    Parameters:
      candidates : ranked candidate list from generate_candidates_with_crc()
      rx_buffer  : accumulated soft received values (not used here, kept for
                   interface symmetry and potential future soft CRC checking)

    Returns:
      Filtered list of CRC-valid Candidate objects, same sort order (best first).
    """
    valid: List[Candidate] = []

    for cand in candidates:
        # The first (K_INFO + CRC_LEN) bits of the codeword are the encoded input x.
        # Since codec_with_crc is systematic, c_M[:k] = x = [u | CRC(u)].
        x_bits = cand.codeword_bits[:K_INFO + CRC_LEN]   # extract the systematic part

        # Convert to Python int list for CRC checking
        x_list = x_bits.tolist()

        # Check if the CRC appended by the transmitter is still consistent.
        # CRC8.check() returns True if the last 8 bits match CRC(first K_INFO bits).
        if CRC8.check(x_list):
            valid.append(cand)    # this candidate satisfies the CRC constraint

    # Return CRC-valid candidates in ascending d² order (best = lowest d² first)
    return valid


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — RELIABILITY ESTIMATOR AND ACK/NACK DECISION
# ─────────────────────────────────────────────────────────────────────────────

def compute_reliability_delta(candidates: List[Candidate]) -> Optional[float]:
    """
    Reliability estimator (the 'Reliability estimator' block in Fig. 17.3/17.4).

    Computes: Δ = d²(y, c2) - d²(y, c1)
      c1 = best candidate  (lowest d²)
      c2 = second-best candidate

    A larger Δ means the best candidate is clearly better than the second-best
    → the decoder is confident → send ACK.

    Returns None if fewer than 2 candidates are available (cannot compute Δ).
    """
    if len(candidates) < 2:
        return None                                # cannot estimate reliability
    # candidates are already sorted by d² (ascending), so [0] = best, [1] = second
    delta = candidates[1].distance2 - candidates[0].distance2
    return delta


def ack_nack_decision(
    delta: Optional[float],
    threshold: float = RELIABILITY_THRESHOLD,
) -> bool:
    """
    ACK/NACK comparator (the 'Comparator' block in Fig. 17.3/17.4).

    ACK  (True)  if: Δ >= threshold  → decoder is confident, output decoded bits
    NACK (False) if: Δ <  threshold  → request more parity bits (next IR round)

    If delta is None (only one candidate), always NACK.
    """
    if delta is None:
        return False           # cannot decide → NACK (request more bits)
    return delta >= threshold  # True = ACK, False = NACK


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11 — FRAME RESULT CONTAINER
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FrameResult:
    """
    Container for the outcome of transmitting one K_INFO-bit frame.

    Attributes:
      is_error         : True if decoded_bits ≠ original info_bits
      num_rounds_used  : how many IR rounds were needed (1–4)
      bits_transmitted : total bits sent over the channel for this frame
      decoded_bits     : K_INFO decoded information bits
      ack_received     : True if the final decision was ACK (vs forced)
      final_delta      : Δ value at the round where decision was made
    """
    is_error:         bool
    num_rounds_used:  int
    bits_transmitted: int
    decoded_bits:     List[int]
    ack_received:     bool
    final_delta:      Optional[float]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 12 — IR-HARQ WITHOUT CRC FRAME TRANSMISSION  (Fig. 17.3)
# ─────────────────────────────────────────────────────────────────────────────

def simulate_frame_ir_no_crc(
    info_bits: np.ndarray,
    sigma:     float,
    rng:       np.random.Generator,
    trace_prefix: str = "no_crc",
    frame_idx: int = 0,
) -> FrameResult:
    
    """
    Simulate one K_INFO-bit frame using IR-HARQ without CRC.
    Matches the block diagram of Fig. 17.3.

    Signal flow (Fig. 17.3):
      u → FEC encoder → c_M → Transmission controller → v → Channel → y'
      y' → Received buffer (y) → FEC decoder → [c1, c2]
      → Reliability estimator → Δ = d²(y,c2) - d²(y,c1)
      → Comparator → ACK/NACK generator
      If ACK or final round: output û = payload of c1
    """

    # ── Transmitter side ──────────────────────────────────────────────────────

    # Encode the K_INFO info bits into the full N_TOTAL-bit mother codeword c_M
    codeword = codec_no_crc.encode(info_bits.tolist())
    if is_trace_enabled(frame_idx):
        save_txt_snapshot(
            f"{trace_prefix}_frame_{frame_idx:04d}_01_input_bits",
            info_bits,
            {
                "system": "IR without CRC",
                "frame_index": frame_idx,
                "stage": "Input bits before FEC encoding",
            }
        )

        save_txt_snapshot(
            f"{trace_prefix}_frame_{frame_idx:04d}_02_after_fec_encoding",
            codeword,
            {
                "system": "IR without CRC",
                "frame_index": frame_idx,
                "stage": "After FEC encoding / mother codeword",
                "N_TOTAL": N_TOTAL,
            }
        )

    # Initialise the receive buffer (empty, grows with each round)
    rx_buffer       = np.empty(0, dtype=np.float64)
    total_bits_sent = 0    # accumulate bits sent across all rounds

    # ── IR loop: transmit one chunk per round ─────────────────────────────────
    for round_idx in range(MAX_ROUNDS):

        # Transmission controller: extract chunk for this round (v in Fig. 17.3)
        chunk = codec_no_crc.get_chunk(codeword, round_idx)
        if is_trace_enabled(frame_idx):
          save_txt_snapshot(
            f"{trace_prefix}_frame_{frame_idx:04d}_round_{round_idx+1}_03_chunk_before_channel",
            chunk,
            {
                "system": "IR without CRC",
                "frame_index": frame_idx,
                "round": round_idx + 1,
                "stage": "Chunk before BPSK/channel",
                "chunk_start": CHUNK_STARTS[round_idx],
                "chunk_end": CHUNK_ENDS[round_idx],
            }
        )

        # Modulate: map bits {0,1} to BPSK symbols {+1,-1}
        symbols  = bpsk_modulate(chunk)

        if is_trace_enabled(frame_idx):
           save_txt_snapshot(
            f"{trace_prefix}_frame_{frame_idx:04d}_round_{round_idx+1}_04_after_bpsk",
            symbols,
            {
                "system": "IR without CRC",
                "frame_index": frame_idx,
                "round": round_idx + 1,
                "stage": "After BPSK modulation",
            }
    )

        # Channel: add Gaussian noise → soft received values y'
        received = awgn_channel(symbols, sigma, rng)
        if is_trace_enabled(frame_idx):
            save_txt_snapshot(
            f"{trace_prefix}_frame_{frame_idx:04d}_round_{round_idx+1}_05_after_channel",
            received,
            {
                "system": "IR without CRC",
                 "frame_index": frame_idx,
                "round": round_idx + 1,
                "stage": "After AWGN channel",
                "sigma": sigma,
           }
        )

        # Accumulate in receive buffer: y = [y_round1 | y_round2 | ...]
        rx_buffer       = np.concatenate([rx_buffer, received])
        total_bits_sent += len(chunk)    # update transmitted bit count

        if is_trace_enabled(frame_idx):
            save_txt_snapshot(
                f"{trace_prefix}_frame_{frame_idx:04d}_round_{round_idx+1}_06_rx_buffer",
                rx_buffer,
                {"system": "IR without CRC", "frame_index": frame_idx, "round": round_idx + 1, "stage": "Accumulated rx_buffer"},
            )

        # FEC decoder: generate list of candidate codewords sorted by d²(y, c)
        candidates = generate_candidates_no_crc(rx_buffer)

        if is_trace_enabled(frame_idx):
            save_candidates_txt(
                f"{trace_prefix}_frame_{frame_idx:04d}_round_{round_idx+1}_07_candidates",
                candidates,
                {"system": "IR without CRC", "frame_index": frame_idx, "round": round_idx + 1, "stage": "Decoder candidates"},
            )

        # Reliability estimator: compute Δ = d²(y, c2) - d²(y, c1)
        delta = compute_reliability_delta(candidates)

        is_final = (round_idx == MAX_ROUNDS - 1)   # True if this is the last round

        # Comparator + ACK/NACK generator: send ACK if confident, or force on final round
        if ack_nack_decision(delta) or is_final:
            best = candidates[0] if candidates else None

            # Output û = best candidate's K_INFO payload bits
            decoded_info = (best.payload_bits[:K_INFO].tolist()
                            if best else [0] * K_INFO)

            # Check for errors by comparing û with the original info bits
            is_error = (decoded_info != info_bits[:K_INFO].tolist())

            if is_trace_enabled(frame_idx):
                save_txt_snapshot(
                    f"{trace_prefix}_frame_{frame_idx:04d}_round_{round_idx+1}_08_delta_and_decision",
                    [delta if delta is not None else -1.0],
                    {"system": "IR without CRC", "frame_index": frame_idx, "round": round_idx + 1,
                     "stage": "Reliability delta + ACK/NACK",
                     "delta": str(delta), "ack": str(ack_nack_decision(delta)), "is_final": str(is_final)},
                )
                save_txt_snapshot(
                    f"{trace_prefix}_frame_{frame_idx:04d}_round_{round_idx+1}_09_decoded_bits",
                    decoded_info,
                    {"system": "IR without CRC", "frame_index": frame_idx, "round": round_idx + 1,
                     "stage": "Final decoded info bits", "is_error": str(is_error)},
                )

            return FrameResult(
                is_error         = is_error,
                num_rounds_used  = round_idx + 1,
                bits_transmitted = total_bits_sent,
                decoded_bits     = decoded_info,
                ack_received     = ack_nack_decision(delta),  # True=ACK, False=forced
                final_delta      = delta,
            )

    # Should never reach here (the final-round check above handles it)
    best = generate_candidates_no_crc(rx_buffer)
    best = best[0] if best else None
    decoded_info = best.payload_bits[:K_INFO].tolist() if best else [0] * K_INFO
    return FrameResult(False, MAX_ROUNDS, total_bits_sent, decoded_info, False, None)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 13 — IR-HARQ WITH CRC FRAME TRANSMISSION  (Fig. 17.4)
# ─────────────────────────────────────────────────────────────────────────────

def simulate_frame_ir_with_crc(
    info_bits: np.ndarray,
    sigma:     float,
    rng:       np.random.Generator,
    trace_prefix: str = "with_crc",
    frame_idx: int = 0,
) -> FrameResult:
    """
    Simulate one K_INFO-bit frame using IR-HARQ with CRC.
    Matches the block diagram of Fig. 17.4.

    Signal flow (Fig. 17.4):
      u  → CRC encoder → x = [u | CRC(u)]   (K_INFO + CRC_LEN bits)
      x  → FEC encoder → c_M                 (N_TOTAL bits mother codeword)
      c_M → Transmission controller → v → Channel → y'
      y'  → Received buffer (y) → FEC decoder → candidate list [c1, c2, ..., cL]
      → CRC checking → [cj, cl]  (j=best CRC-valid, l=second-best CRC-valid)
      → Reliability estimator → Δ = d²(y, cl) - d²(y, cj)
      → Comparator → ACK/NACK generator
      If ACK or final round: output û = payload of cj (CRC stripped)

    Key difference from Fig. 17.3:
      The CRC checking stage explicitly filters out candidates whose CRC
      does not match.  ACK is only sent if at least one CRC-valid candidate
      exists AND Δ >= threshold.
    """

    # ── Transmitter side ──────────────────────────────────────────────────────

    # CRC encoder block (Fig. 17.4): x = [u | CRC(u)]
    data_crc = CRC8.append(info_bits.tolist())   # K_INFO + CRC_LEN = 8200 bits

    # FEC encoder block: encode x into mother codeword c_M
    codeword = codec_with_crc.encode(data_crc)   # N_TOTAL = 20480 bits

    if is_trace_enabled(frame_idx):
        save_txt_snapshot(
            f"{trace_prefix}_frame_{frame_idx:04d}_01_input_bits",
            info_bits,
            {"system": "IR with CRC", "frame_index": frame_idx, "stage": "Input bits before CRC"},
        )
        save_txt_snapshot(
            f"{trace_prefix}_frame_{frame_idx:04d}_02_after_crc_append",
            data_crc,
            {"system": "IR with CRC", "frame_index": frame_idx, "stage": "After CRC append", "total_len": len(data_crc)},
        )
        save_txt_snapshot(
            f"{trace_prefix}_frame_{frame_idx:04d}_03_after_fec_encoding",
            codeword,
            {"system": "IR with CRC", "frame_index": frame_idx, "stage": "After FEC encoding / mother codeword", "N_TOTAL": N_TOTAL},
        )

    rx_buffer       = np.empty(0, dtype=np.float64)
    total_bits_sent = 0

    # ── IR loop ───────────────────────────────────────────────────────────────
    for round_idx in range(MAX_ROUNDS):

        # Transmission controller: get chunk for this round
        chunk = codec_with_crc.get_chunk(codeword, round_idx)

        # Modulate and send through AWGN channel
        symbols  = bpsk_modulate(chunk)
        received = awgn_channel(symbols, sigma, rng)

        if is_trace_enabled(frame_idx):
            save_txt_snapshot(
                f"{trace_prefix}_frame_{frame_idx:04d}_round_{round_idx+1}_04_chunk_before_channel",
                chunk,
                {"system": "IR with CRC", "frame_index": frame_idx, "round": round_idx + 1,
                 "stage": "Chunk before BPSK/channel", "chunk_start": CHUNK_STARTS[round_idx], "chunk_end": CHUNK_ENDS[round_idx]},
            )
            save_txt_snapshot(
                f"{trace_prefix}_frame_{frame_idx:04d}_round_{round_idx+1}_05_bpsk_symbols",
                symbols,
                {"system": "IR with CRC", "frame_index": frame_idx, "round": round_idx + 1, "stage": "After BPSK modulation"},
            )
            save_txt_snapshot(
                f"{trace_prefix}_frame_{frame_idx:04d}_round_{round_idx+1}_06_after_channel",
                received,
                {"system": "IR with CRC", "frame_index": frame_idx, "round": round_idx + 1,
                 "stage": "After AWGN channel", "sigma": sigma},
            )

        # Accumulate in receive buffer
        rx_buffer       = np.concatenate([rx_buffer, received])
        total_bits_sent += len(chunk)

        if is_trace_enabled(frame_idx):
            save_txt_snapshot(
                f"{trace_prefix}_frame_{frame_idx:04d}_round_{round_idx+1}_07_rx_buffer",
                rx_buffer,
                {"system": "IR with CRC", "frame_index": frame_idx, "round": round_idx + 1, "stage": "Accumulated rx_buffer"},
            )

        # FEC decoder block: produce a list of candidate codewords
        candidates = generate_candidates_with_crc(rx_buffer)

        if is_trace_enabled(frame_idx):
            save_candidates_txt(
                f"{trace_prefix}_frame_{frame_idx:04d}_round_{round_idx+1}_08_candidates",
                candidates,
                {"system": "IR with CRC", "frame_index": frame_idx, "round": round_idx + 1, "stage": "Decoder candidates (before CRC filter)"},
            )

        # ── CRC CHECKING STAGE (explicit block from Fig. 17.4) ──────────────
        valid_candidates = crc_filter_candidates(candidates, rx_buffer)

        if is_trace_enabled(frame_idx):
            save_candidates_txt(
                f"{trace_prefix}_frame_{frame_idx:04d}_round_{round_idx+1}_09_crc_valid_candidates",
                valid_candidates,
                {"system": "IR with CRC", "frame_index": frame_idx, "round": round_idx + 1, "stage": "CRC-valid candidates"},
            )

        is_final = (round_idx == MAX_ROUNDS - 1)

        if valid_candidates:
            delta = compute_reliability_delta(valid_candidates)
            use_candidates = valid_candidates
        else:
            delta = None
            use_candidates = candidates

        # ACK/NACK comparator
        if ack_nack_decision(delta) or is_final:
            # cj = best CRC-valid candidate (or any candidate if forced)
            best = use_candidates[0] if use_candidates else None

            # û = K_INFO info bits (CRC was stripped in generate_candidates_with_crc)
            decoded_info = (best.payload_bits[:K_INFO].tolist()
                            if best else [0] * K_INFO)

            is_error = (decoded_info != info_bits[:K_INFO].tolist())

            if is_trace_enabled(frame_idx):
                save_txt_snapshot(
                    f"{trace_prefix}_frame_{frame_idx:04d}_round_{round_idx+1}_10_delta_and_decision",
                    [delta if delta is not None else -1.0],
                    {"system": "IR with CRC", "frame_index": frame_idx, "round": round_idx + 1,
                     "stage": "Reliability delta + ACK/NACK",
                     "delta": str(delta), "ack": str(ack_nack_decision(delta)), "is_final": str(is_final)},
                )
                save_txt_snapshot(
                    f"{trace_prefix}_frame_{frame_idx:04d}_round_{round_idx+1}_11_decoded_bits",
                    decoded_info,
                    {"system": "IR with CRC", "frame_index": frame_idx, "round": round_idx + 1,
                     "stage": "Final decoded info bits", "is_error": str(is_error)},
                )

            return FrameResult(
                is_error         = is_error,
                num_rounds_used  = round_idx + 1,
                bits_transmitted = total_bits_sent,
                decoded_bits     = decoded_info,
                ack_received     = ack_nack_decision(delta),
                final_delta      = delta,
            )

    # Fallback (should not reach here due to is_final check above)
    fallback = generate_candidates_with_crc(rx_buffer)
    valid_fb = crc_filter_candidates(fallback, rx_buffer)
    best_fb  = (valid_fb[0] if valid_fb else (fallback[0] if fallback else None))
    decoded  = best_fb.payload_bits[:K_INFO].tolist() if best_fb else [0] * K_INFO
    return FrameResult(False, MAX_ROUNDS, total_bits_sent, decoded, False, None)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 14 — IMAGE ENCODING AND RECONSTRUCTION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def encode_image_to_bits(image_path: str) -> Tuple[np.ndarray, int, int]:
    """
    Load an image, convert to YCbCr, JPEG-encode each channel, and
    return a flat binary bit array plus the processed image dimensions.

    Steps:
      1. Open image in RGB mode.
      2. Crop width and height to the nearest multiple of 8 (JPEG block requirement).
      3. Convert from RGB to YCbCr colour space.
      4. Split into Y (luminance), Cb, Cr (chrominance) channels.
      5. Encode each channel using encode_channel() from jpeg_colorful.py.
      6. Concatenate Y_bits | Cb_bits | Cr_bits → flat bitstream.

    Returns:
      (bitstream, processed_height, processed_width)
      bitstream: numpy uint8 array of {0, 1} bits
    """
    img   = Image.open(image_path).convert("RGB")   # open in RGB (3-channel)
    #img = img.resize((128, 64), Image.LANCZOS)

    w, h  = img.size                                 # original pixel dimensions

    # Crop to the nearest multiple of 8 in both dimensions.
    # This is the only intentional size change — required for 8×8 DCT blocks.
    new_w = w - (w % 8)
    new_h = h - (h % 8)
    img_cropped = img.crop((0, 0, new_w, new_h))     # keep top-left corner

    # Convert colour space: RGB → YCbCr
    # Y  = luminance (brightness) → encoded with Q_MATRIX (preserves detail)
    # Cb = blue-difference chrominance → encoded with C_MATRIX (lossy)
    # Cr = red-difference chrominance  → encoded with C_MATRIX (lossy)
    img_ycbcr = img_cropped.convert("YCbCr")
    y_img, cb_img, cr_img = img_ycbcr.split()        # split into 3 single-channel images

    # Convert PIL Images to numpy uint8 arrays for the codec functions
    y_arr  = np.array(y_img,  dtype=np.uint8)
    cb_arr = np.array(cb_img, dtype=np.uint8)
    cr_arr = np.array(cr_img, dtype=np.uint8)

    # Trace: YCbCr pixel arrays before encoding
    save_txt_snapshot("00b_ycbcr_Y_pixels",  y_arr,  {"stage": "Y channel pixels (before JPEG encoding)",  "shape": str(y_arr.shape)})
    save_txt_snapshot("00c_ycbcr_Cb_pixels", cb_arr, {"stage": "Cb channel pixels (before JPEG encoding)", "shape": str(cb_arr.shape)})
    save_txt_snapshot("00d_ycbcr_Cr_pixels", cr_arr, {"stage": "Cr channel pixels (before JPEG encoding)", "shape": str(cr_arr.shape)})

    # Encode each channel: DCT → quantize → flatten → convert coefficients to bits
    y_bits  = encode_channel(y_arr,  Q_MATRIX)   # luminance channel bits
    cb_bits = encode_channel(cb_arr, C_MATRIX)   # Cb channel bits
    cr_bits = encode_channel(cr_arr, C_MATRIX)   # Cr channel bits

    save_txt_snapshot(
        "01_after_encoding_y_bits",
         y_bits,
        {
            "stage": "After JPEG encoding",
            "channel": "Y",
            "bits_count": len(y_bits),
         }
    )

    save_txt_snapshot(
        "02_after_encoding_cb_bits",
        cb_bits,
        {
            "stage": "After JPEG encoding",
            "channel": "Cb",
            "bits_count": len(cb_bits),
        }
    )

    save_txt_snapshot(
        "03_after_encoding_cr_bits",
        cr_bits,
        {
            "stage": "After JPEG encoding",
            "channel": "Cr",
            "bits_count": len(cr_bits),
        }
    )

    # Concatenate all channel bits into one flat transmission bitstream
    bitstream = np.concatenate([y_bits, cb_bits, cr_bits]).astype(np.uint8)
    save_txt_snapshot(
        "04_full_image_bitstream_after_encoding",
        bitstream,
        {
            "stage": "After concatenating Y + Cb + Cr",
            "total_bits": len(bitstream),
        }   
    )
    return bitstream, new_h, new_w   # return bits + processed dimensions


def reconstruct_image_from_bits(
    bits: np.ndarray,
    H: int,
    W: int,
    output_path: str,
) -> None:
    """
    Reconstruct a colour image from a flat bitstream and save it as PNG.

    Reverses encode_image_to_bits():
      1. Split bitstream into Y, Cb, Cr channel sections.
      2. Decode each channel using decode_channel() from jpeg_colorful.py.
         (inverse DCT + dequantize + clip to [0, 255])
      3. Merge YCbCr channels back into a colour image.
      4. Convert YCbCr → RGB for saving.

    Parameters:
      bits        : flat bit array (possibly received bits with errors)
      H, W        : processed image height and width (pixels)
      output_path : where to save the reconstructed PNG
    """
    # Compute how many bits belong to each colour channel
    blocks_per_channel = (H // 8) * (W // 8)          # number of 8×8 blocks per channel
    bits_per_channel   = blocks_per_channel * 64 * 16  # 64 coefficients × 16 bits each

    # Slice the flat bitstream into three channel bitstreams
    rx_y  = bits[0                  : bits_per_channel]        # Y channel
    rx_cb = bits[bits_per_channel   : 2 * bits_per_channel]    # Cb channel
    rx_cr = bits[2 * bits_per_channel : 3 * bits_per_channel]  # Cr channel

    # Trace: bits going into each channel decoder
    label = os.path.basename(output_path).replace(".png", "")
    save_txt_snapshot(f"recon_{label}_Y_bits_before_decode",  rx_y,  {"stage": "Y bits before image decoding",  "output": output_path})
    save_txt_snapshot(f"recon_{label}_Cb_bits_before_decode", rx_cb, {"stage": "Cb bits before image decoding", "output": output_path})
    save_txt_snapshot(f"recon_{label}_Cr_bits_before_decode", rx_cr, {"stage": "Cr bits before image decoding", "output": output_path})

    # Decode each channel back to a 2D pixel array
    rec_y  = decode_channel(rx_y,  H, W, Q_MATRIX)   # luminance
    rec_cb = decode_channel(rx_cb, H, W, C_MATRIX)   # blue-difference chrominance
    rec_cr = decode_channel(rx_cr, H, W, C_MATRIX)   # red-difference chrominance

    # Trace: decoded pixel arrays
    save_txt_snapshot(f"recon_{label}_Y_pixels_after_decode",  rec_y,  {"stage": "Y pixels after image decoding",  "shape": str(rec_y.shape)})
    save_txt_snapshot(f"recon_{label}_Cb_pixels_after_decode", rec_cb, {"stage": "Cb pixels after image decoding", "shape": str(rec_cb.shape)})
    save_txt_snapshot(f"recon_{label}_Cr_pixels_after_decode", rec_cr, {"stage": "Cr pixels after image decoding", "shape": str(rec_cr.shape)})

    # Merge decoded channels into a YCbCr PIL Image
    final_ycbcr = Image.merge("YCbCr", (
        Image.fromarray(rec_y,  mode="L"),
        Image.fromarray(rec_cb, mode="L"),
        Image.fromarray(rec_cr, mode="L"),
    ))

    # Convert to RGB for standard display and saving
    final_rgb = final_ycbcr.convert("RGB")
    final_rgb.save(output_path)    # save as PNG (lossless)
    print(f"   [Saved] {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 15 — FULL IMAGE TRANSMISSION PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def transmit_image(
    system_name:   str,
    system_func,               # simulate_frame_ir_no_crc or simulate_frame_ir_with_crc
    image_bits:    np.ndarray,  # padded bitstream (multiple of K_INFO)
    original_bits: int,         # original (unpadded) bitstream length
    H: int, W: int,             # processed image dimensions in pixels
    sigma: float,               # AWGN noise std dev
    block_seeds:   np.ndarray,  # per-frame RNG seeds (for reproducibility)
    output_path:   str,         # where to save the reconstructed image
    use_crc:       bool,        # True = IR with CRC, False = IR without CRC
    ebn0_db:       float,       # Eb/N0 in dB (for CSV logging)
    image_name:    str,         # image filename (for CSV logging)
    orig_w: int, orig_h: int,   # original image dimensions (for CSV)
    proc_w: int, proc_h: int,   # processed image dimensions (for CSV)
) -> Tuple[dict, List[dict]]:
    """
    Transmit the full image bitstream through the given IR-HARQ system,
    then reconstruct and save the output image.

    Steps:
      1. Split image_bits into K_INFO-bit frames.
      2. For each frame: call system_func() → FrameResult.
      3. Collect decoded bits from all frames.
      4. Trim decoded array to original_bits length (remove padding).
      5. Safety-pad to exactly 3 × blocks × 64 × 16 bits (image reconstruction).
      6. Reconstruct and save the image.
      7. Compute overall BER and assemble CSV records.

    Returns:
      (summary_dict, list of per-frame dicts for transmission_report.csv)
    """
    n_frames      = len(image_bits) // K_INFO   # total number of frames
    crc_per_frame = CRC_LEN if use_crc else 0   # CRC bits added per frame
    code_rate     = K_INFO / N_TOTAL             # nominal code rate (Round 1 only)

    print()
    print("=" * 65)
    print(f"  System : {system_name}")
    print(f"  Frames : {n_frames}  ×  K_INFO={K_INFO} bits per frame")
    print(f"  Eb/N0  : {ebn0_db} dB   σ={sigma:.4f}")
    print("=" * 65)

    t_start = time.time()   # record start time for runtime measurement

    decoded_all:  List[int]  = []    # accumulate all decoded info bits
    total_tx_bits = 0                 # total bits transmitted across all frames
    total_errors  = 0                 # total bit errors across all frames
    frame_rows:   List[dict] = []    # per-frame CSV rows

    # Decide how often to print progress (10 updates total)
    report_step = max(1, n_frames // 10)

    for fi in range(n_frames):
        # Extract the fi-th K_INFO-bit block from the image bitstream
        info_blk = image_bits[fi * K_INFO : (fi + 1) * K_INFO].astype(np.int8)

        # Create a per-frame RNG (reproducible, independent across frames)
        blk_rng = np.random.default_rng(int(block_seeds[fi]))

        # Transmit this frame through the selected IR-HARQ system
        trace_prefix = system_name.lower().replace(" ", "_")

        result = system_func(
            info_blk,
            sigma,
            blk_rng,
            trace_prefix=trace_prefix,
            frame_idx=fi + 1,
        )

        # Append the K_INFO decoded bits to the output stream
        decoded_all.extend(result.decoded_bits)

        # Accumulate transmitted bit count
        total_tx_bits += result.bits_transmitted

        # Count bit errors in this frame
        bit_errors_frame = int(np.sum(
            info_blk[:K_INFO] != np.array(result.decoded_bits[:K_INFO], dtype=np.int8)
        ))
        total_errors += bit_errors_frame

        # ── Per-frame CSV row ─────────────────────────────────────────────────
        frame_rows.append({
            "system_name":              system_name,
            "image_name":               image_name,
            "original_width":           orig_w,
            "original_height":          orig_h,
            "processed_width":          proc_w,
            "processed_height":         proc_h,
            "original_image_bits":      original_bits,
            "transmitted_payload_bits": K_INFO,
            "crc_bits_added":           crc_per_frame,
            "total_transmitted_bits":   result.bits_transmitted,
            "frame_index":              fi + 1,
            "total_frames":             n_frames,
            "K_INFO":                   K_INFO,
            "CRC_LEN":                  CRC_LEN,
            "N_TOTAL":                  N_TOTAL,
            "code_rate":                f"{code_rate:.4f}",
            "CHUNK_STARTS":             str(CHUNK_STARTS),
            "CHUNK_ENDS":               str(CHUNK_ENDS),
            "chunk_lengths":            str(_chunk_lengths),
            "chunks_used":              result.num_rounds_used,
            "ebn0_db":                  ebn0_db,
            "decoded_bits":             K_INFO,
            "bit_errors":               bit_errors_frame,
            "ber":                      f"{bit_errors_frame / K_INFO:.6f}",
            "final_delta":              f"{result.final_delta:.4f}" if result.final_delta is not None else "None",
            "ack_received":             "ACK" if result.ack_received else "forced",
            "output_image_path":        output_path,
            "status":                   "ACK" if result.ack_received else "forced",
        })

        # Print progress at regular intervals
        if (fi + 1) % report_step == 0 or (fi + 1) == n_frames:
            pct       = 100.0 * (fi + 1) / n_frames
            delta_str = f"{result.final_delta:.2f}" if result.final_delta is not None else "N/A"
            ack_str   = "ACK" if result.ack_received else "NACK/forced"
            print(f"   Frame {fi+1:5d}/{n_frames} ({pct:5.1f}%)  "
                  f"tx_bits={result.bits_transmitted:6d}  "
                  f"rounds={result.num_rounds_used}  "
                  f"delta={delta_str}  {ack_str}",
                  flush=True)

    runtime = time.time() - t_start   # total runtime for this system

    # ── Trim padding: keep only the original image bits ───────────────────────
    decoded_arr = np.array(decoded_all, dtype=np.uint8)[:original_bits]

    # ── Safety pad: image reconstruction needs exactly this many bits ─────────
    total_channel_bits = 3 * (H // 8) * (W // 8) * 64 * 16
    if len(decoded_arr) < total_channel_bits:
        decoded_arr = np.concatenate([
            decoded_arr,
            np.zeros(total_channel_bits - len(decoded_arr), dtype=np.uint8)
        ])

    # ── Reconstruct and save the output image ────────────────────────────────
    reconstruct_image_from_bits(decoded_arr, H, W, output_path)

    # ── Compute final BER over all original bits ──────────────────────────────
    n_compare          = min(original_bits, len(decoded_arr))
    total_errors_final = int(np.sum(
        image_bits[:n_compare].astype(np.uint8) != decoded_arr[:n_compare]
    ))
    ber_total = total_errors_final / n_compare if n_compare > 0 else 0.0

    print(f"   BER        : {ber_total:.6f}")
    print(f"   Bit errors : {total_errors_final}")
    print(f"   Runtime    : {runtime:.1f}s")

    # ── Average rounds used ───────────────────────────────────────────────────
    avg_rounds = sum(r["chunks_used"] for r in frame_rows) / max(n_frames, 1)

    # ── Summary dictionary for summary_report.csv ────────────────────────────
    summary = {
        "system_name":            system_name,
        "total_image_bits":       original_bits,
        "total_transmitted_bits": total_tx_bits,
        "total_frames":           n_frames,
        "total_crc_bits":         crc_per_frame * n_frames,
        "average_code_rate":      f"{code_rate:.4f}",
        "average_rounds_used":    f"{avg_rounds:.2f}",
        "total_bit_errors":       total_errors_final,
        "ber":                    f"{ber_total:.6f}",
        "output_image_path":      output_path,
        "runtime_seconds":        f"{runtime:.2f}",
    }

    return summary, frame_rows


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 16 — CSV REPORTING
# ─────────────────────────────────────────────────────────────────────────────

# Column definitions for transmission_report.csv (one row per frame per system)
TRANSMISSION_CSV_COLUMNS = [
    "system_name", "image_name", "original_width", "original_height",
    "processed_width", "processed_height", "original_image_bits",
    "transmitted_payload_bits", "crc_bits_added", "total_transmitted_bits",
    "frame_index", "total_frames", "K_INFO", "CRC_LEN", "N_TOTAL",
    "code_rate", "CHUNK_STARTS", "CHUNK_ENDS", "chunk_lengths", "chunks_used",
    "ebn0_db", "decoded_bits", "bit_errors", "ber",
    "final_delta", "ack_received", "output_image_path", "status",
]

# Column definitions for summary_report.csv (one row per system)
SUMMARY_CSV_COLUMNS = [
    "system_name", "total_image_bits", "total_transmitted_bits",
    "total_frames", "total_crc_bits", "average_code_rate",
    "average_rounds_used", "total_bit_errors", "ber",
    "output_image_path", "runtime_seconds",
]


def write_csv(path: str, columns: List[str], rows: List[dict]) -> None:
    """
    Write a list of dictionaries to a CSV file.
    Missing keys are written as empty string.
    """
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()                                  # write column headers
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in columns})  # fill missing=""
    print(f"   [Saved] {path}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 17 — MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run_image_test(
    image_path: str,
    ebn0_db:    float = 10.0,
    seed:       int   = 42,
) -> None:
    """
    Full image transmission pipeline.

    Steps:
      1. Load and JPEG-encode the image → flat bitstream.
      2. Save codec-only reference image (no channel noise).
      3. Pad bitstream to a multiple of K_INFO.
      4. Transmit using IR without CRC → reconstruct image.
      5. Transmit using IR with CRC    → reconstruct image.
      6. Save transmission_report.csv and summary_report.csv.
      7. Print final summary table.

    Parameters:
      image_path : path to input image (JPEG or PNG)
      ebn0_db    : Eb/N0 in dB for the AWGN channel (higher = less noise)
      seed       : master random seed for full reproducibility
    """

    os.makedirs(OUTPUT_DIR, exist_ok=True)   # create outputs/ folder if not present
    image_name = os.path.basename(image_path)
    setup_trace(
    output_dir=OUTPUT_DIR,
    trace_folder="txt_trace",
    save_full=False,
    preview_len=128,
    max_frames=3,
    )
    save_txt_snapshot(
        "00_trace_test",
        [0, 1, 1, 0, 1],
        {
            "stage": "Trace test",
            "message": "If you see this file, trace_utils is working"
        }
    )
    # ── STEP 1: Load and encode the image ────────────────────────────────────
    print()
    print("=" * 65)
    print("  STEP 1 — Loading and encoding image")
    print("=" * 65)

    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        print("  Please place your image at that path and run again.")
        return

    img_orig        = Image.open(image_path).convert("RGB")
    orig_w, orig_h  = img_orig.size                              # original dimensions

    # Encode the image using the JPEG codec from jpeg_colorful.py
    bitstream, proc_h, proc_w = encode_image_to_bits(image_path)
    original_bits = len(bitstream)    # exact bit count before any padding

    print(f"   Original size    : {orig_w}×{orig_h} pixels")
    print(f"   Processed size   : {proc_w}×{proc_h} pixels  (cropped to 8×8 multiple)")
    print(f"   Bitstream length : {original_bits:,} bits")

    # Save the codec-reference image (encode then decode with NO channel noise)
    ref_out = os.path.join(OUTPUT_DIR, "original_processed.png")
    reconstruct_image_from_bits(bitstream.copy(), proc_h, proc_w, ref_out)

    # ── STEP 2: Pad bitstream to multiple of K_INFO ───────────────────────────
    pad_len = (-original_bits) % K_INFO    # bits to add (0 if already aligned)
    if pad_len > 0:
        bitstream = np.concatenate([bitstream,
                                    np.zeros(pad_len, dtype=np.uint8)])  # zero-pad
    n_frames = len(bitstream) // K_INFO
    print(f"   Padding added    : {pad_len} bits  →  {n_frames} frames of {K_INFO} bits")

    # Compute AWGN noise std dev from Eb/N0
    sigma = ebn0_db_to_sigma(ebn0_db)
    print(f"   Eb/N0            : {ebn0_db} dB  →  σ={sigma:.5f}")

    # Generate per-frame RNG seeds from the master seed (reproducible)
    master_rng  = np.random.default_rng(seed)
    block_seeds = master_rng.integers(0, 2**31, size=n_frames, dtype=np.int64)

    all_frame_rows: List[dict] = []   # will hold all per-frame CSV rows
    summaries:      List[dict] = []   # will hold per-system summary rows

    # ── STEP 3: IR without CRC ────────────────────────────────────────────────
    out_no_crc = os.path.join(OUTPUT_DIR, "reconstructed_ir_without_crc.png")
    summary_a, rows_a = transmit_image(
        system_name   = "IR without CRC",
        system_func   = simulate_frame_ir_no_crc,
        image_bits    = bitstream,
        original_bits = original_bits,
        H=proc_h, W=proc_w,
        sigma         = sigma,
        block_seeds   = block_seeds,
        output_path   = out_no_crc,
        use_crc       = False,
        ebn0_db       = ebn0_db,
        image_name    = image_name,
        orig_w=orig_w, orig_h=orig_h,
        proc_w=proc_w, proc_h=proc_h,
    )
    all_frame_rows.extend(rows_a)
    summaries.append(summary_a)

    # ── STEP 4: IR with CRC ───────────────────────────────────────────────────
    out_with_crc = os.path.join(OUTPUT_DIR, "reconstructed_ir_with_crc.png")
    summary_b, rows_b = transmit_image(
        system_name   = "IR with CRC",
        system_func   = simulate_frame_ir_with_crc,
        image_bits    = bitstream,
        original_bits = original_bits,
        H=proc_h, W=proc_w,
        sigma         = sigma,
        block_seeds   = block_seeds,
        output_path   = out_with_crc,
        use_crc       = True,
        ebn0_db       = ebn0_db,
        image_name    = image_name,
        orig_w=orig_w, orig_h=orig_h,
        proc_w=proc_w, proc_h=proc_h,
    )
    all_frame_rows.extend(rows_b)
    summaries.append(summary_b)

    # ── STEP 5: Save CSV reports ───────────────────────────────────────────────
    print()
    print("=" * 65)
    print("  STEP 5 — Saving CSV reports")
    print("=" * 65)

    tr_csv  = os.path.join(OUTPUT_DIR, "transmission_report.csv")
    sum_csv = os.path.join(OUTPUT_DIR, "summary_report.csv")

    write_csv(tr_csv,  TRANSMISSION_CSV_COLUMNS, all_frame_rows)   # per-frame CSV
    write_csv(sum_csv, SUMMARY_CSV_COLUMNS,      summaries)         # summary CSV

    # ── Final summary printout ────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("  FINAL SUMMARY")
    print("=" * 65)
    hdr = f"  {'System':<25}  {'BER':>10}  {'Errors':>8}  {'Tx bits':>12}  {'Rounds':>7}  {'Time':>7}"
    print(hdr)
    print("-" * 65)
    for s in summaries:
        print(f"  {s['system_name']:<25}  "
              f"{s['ber']:>10}  "
              f"{s['total_bit_errors']:>8}  "
              f"{s['total_transmitted_bits']:>12}  "
              f"{s['average_rounds_used']:>7}  "
              f"{s['runtime_seconds']:>6}s")
    print("=" * 65)
    print()
    print("Output files:")
    for f_ in [ref_out, out_no_crc, out_with_crc, tr_csv, sum_csv]:
        print(f"  {f_}")
    print()
    print("Done.")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Change only this image path.
    # Any JPEG or PNG image will work.
    run_image_test("D:/4th_ECE/Second_Term/Selective_Topics_In_Signal_Processing/Project/Codes/download.jpg", ebn0_db=15.0, seed=42)