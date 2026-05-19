import numpy as np

def get_dct_basis(n):
    """
    Generates the DCT-II basis matrix of size n.
    """
    basis = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == 0:
                basis[i, j] = 1.0 / np.sqrt(n)
            else:
                basis[i, j] = np.sqrt(2.0 / n) * np.cos((np.pi * i * (2 * j + 1)) / (2.0 * n))
    return basis

# Precompute basis for common block sizes to keep it fast
BASIS_8x8 = get_dct_basis(8)
BASIS_16x16 = get_dct_basis(16)

def block_dct(block):
    """
    Computes the 2D Discrete Cosine Transform for an NxN macroblock.
    Formula: D = B * block * B.T
    where B is the DCT basis matrix.
    Pedagogical Note: DCT transforms pixels from the spatial domain to the frequency domain.
    Most of the 'energy' of the image is packed into the low-frequency coefficients (top-left).
    """
    n = block.shape[0]
    if n == 8:
        basis = BASIS_8x8
    elif n == 16:
        basis = BASIS_16x16
    else:
        basis = get_dct_basis(n)

    # 2D DCT using matrix multiplication: B * block * B^T
    return np.dot(np.dot(basis, block), basis.T)

def block_idct(block):
    """
    Computes the inverse 2D DCT.
    Formula: block = B.T * D * B
    Reconstructs the spatial domain pixels from the frequency domain coefficients.
    """
    n = block.shape[0]
    if n == 8:
        basis = BASIS_8x8
    elif n == 16:
        basis = BASIS_16x16
    else:
        basis = get_dct_basis(n)

    # 2D Inverse DCT: B^T * block * B
    return np.dot(np.dot(basis.T, block), basis)

# Standard JPEG Luminance Quantization Table
JPEG_LUM_QUANT = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
], dtype=np.float32)

def get_quantization_matrix(qp, matrix_type='flat', block_size=8):
    """
    Returns a quantization matrix based on QP and type.
    qp: 1 to 100 (where 1 is highest quality, 100 is lowest quality in this implementation's scale)
    """
    if matrix_type == 'weighted' and block_size == 8:
        # Scale the JPEG matrix by QP
        # In real JPEG, quality 50 means scale 1.0. Higher QP means lower quality.
        # Let's map QP 50 to scale 1.0.
        scale = qp / 50.0
        return np.maximum(1, JPEG_LUM_QUANT * scale)
    else:
        # Flat matrix: all coefficients are treated equally.
        # Pedagogical Note: Flat matrices are simpler but don't exploit the fact that
        # human eyes are less sensitive to high-frequency artifacts.
        return np.ones((block_size, block_size)) * qp

def quantize(dct_block, qp, matrix_type='flat'):
    """
    Quantizes the DCT coefficients.
    Pedagogical Note: This is the ONLY lossy step in video compression.
    By dividing the coefficients by a quantization value and rounding to the nearest integer,
    we discard 'unimportant' details. High QP = more zeros = better compression = lower quality.
    """
    q_matrix = get_quantization_matrix(qp, matrix_type, dct_block.shape[0])
    return np.round(dct_block / q_matrix)

def dequantize(quantized_block, qp, matrix_type='flat'):
    """
    Reconstructs the DCT coefficients by multiplying by the quantization matrix.
    Note: The values lost during rounding in the quantization step cannot be recovered.
    """
    q_matrix = get_quantization_matrix(qp, matrix_type, quantized_block.shape[0])
    return quantized_block * q_matrix

def motion_search(current_block, reference_frame, block_pos, search_window):
    """
    Implements Exhaustive Search block-matching.
    current_block: The macroblock from the current frame.
    reference_frame: The previous (or future) reconstructed frame.
    block_pos: (y, x) coordinates of the current block.
    search_window: How many pixels to search in each direction.

    Returns: (best_mv, predicted_block)
    """
    y, x = block_pos
    h, w = current_block.shape
    ref_h, ref_w = reference_frame.shape

    best_sad = float('inf')
    best_mv = (0, 0)

    # Define search range boundaries
    y_min = max(0, y - search_window)
    y_max = min(ref_h - h, y + search_window)
    x_min = max(0, x - search_window)
    x_max = min(ref_w - w, x + search_window)

    # Pedagogical Note: Exhaustive search checks every possible position in the window.
    # It is slow but guaranteed to find the absolute best match (lowest SAD).
    for ry in range(y_min, y_max + 1):
        for rx in range(x_min, x_max + 1):
            ref_block = reference_frame[ry:ry+h, rx:rx+w]
            # Sum of Absolute Differences (SAD)
            sad = np.sum(np.abs(current_block.astype(np.int16) - ref_block.astype(np.int16)))

            if sad < best_sad:
                best_sad = sad
                best_mv = (ry - y, rx - x)

    predicted_block = reference_frame[y+best_mv[0]:y+best_mv[0]+h, x+best_mv[1]:x+best_mv[1]+w]
    return best_mv, predicted_block

def compute_residual(current_block, predicted_block):
    """
    Calculates the difference (residual) between the actual block and the predicted block.
    Pedagogical Note: If motion estimation is good, the residual will be mostly zeros/small values,
    which compress much better than the original image data.
    """
    return current_block.astype(np.float32) - predicted_block.astype(np.float32)

def reconstruct_macroblock(predicted_block, residual_block):
    """
    Rebuilds the block by adding the residual back to the prediction.
    """
    reconstructed = predicted_block.astype(np.float32) + residual_block.astype(np.float32)
    return np.clip(reconstructed, 0, 255).astype(np.uint8)

def parse_gop_structure(gop_string, num_frames):
    """
    Assigns frame types (I, P, B) to the video sequence based on a GOP pattern.
    Example: 'IBBP' for 8 frames -> 'IBBPIBBP'
    """
    if not gop_string:
        return ['I'] * num_frames

    types = []
    for i in range(num_frames):
        types.append(gop_string[i % len(gop_string)])
    return types

def auto_gop_detect(frame_sequence, threshold=30.0):
    """
    Automatically inserts an I-frame if the scene change (average pixel difference)
    exceeds a threshold.
    """
    types = ['I'] # First frame is always I
    if len(frame_sequence) < 2:
        return types

    for i in range(1, len(frame_sequence)):
        diff = np.mean(np.abs(frame_sequence[i].astype(np.float32) - frame_sequence[i-1].astype(np.float32)))
        if diff > threshold:
            types.append('I')
        else:
            types.append('P')
    return types

def estimate_frame_bits(quantized_blocks):
    """
    A very rough estimation of bitrate/size based on non-zero coefficients.
    In a real codec, this would involve Entropy Coding (Huffman/CABAC).
    """
    # Count non-zero coefficients as a proxy for bits
    non_zeros = np.count_nonzero(quantized_blocks)
    # Assume roughly 8 bits per non-zero coefficient + some overhead
    return non_zeros * 8

def encode_vbr(target_qp):
    """
    Variable Bitrate strategy.
    Returns the QP to use (constant).
    """
    return target_qp

def encode_cbr(target_bitrate_per_frame, current_estimated_bits, current_qp):
    """
    Constant Bitrate strategy (Simple Rate Control).
    Adjusts QP based on whether we are over or under the bit budget for the frame.
    """
    if current_estimated_bits > target_bitrate_per_frame:
        return min(current_qp + 2, 100)
    elif current_estimated_bits < target_bitrate_per_frame:
        return max(current_qp - 2, 1)
    return current_qp

def calculate_psnr(original, reconstructed):
    """
    Calculates Peak Signal-to-Noise Ratio.
    Higher PSNR usually means better quality.
    """
    mse = np.mean((original.astype(np.float32) - reconstructed.astype(np.float32)) ** 2)
    if mse < 1e-10:
        return 100.0
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr
