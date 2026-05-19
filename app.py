import streamlit as st
import cv2
import numpy as np
import codec_engine as ce
import tempfile
import time
import os
from PIL import Image

st.set_page_config(page_title="Educational Video Codec", layout="wide")

st.title("🎥 Educational Video Codec: Under the Hood")
st.markdown("""
This tool demonstrates how video compression works by breaking down the process into
Transform (DCT), Quantization, and Motion Compensation.
""")

# --- Sidebar Controls ---
st.sidebar.header("Codec Settings")
codec_mode = st.sidebar.selectbox("Codec Mode", ["MJPEG (Intra-only)", "MPEG (Intra + Inter)"])

gop_auto = st.sidebar.toggle("Auto Scene Detect", value=False)
if not gop_auto:
    gop_structure = st.sidebar.text_input("GOP Structure", value="IPPP")
else:
    scene_threshold = st.sidebar.slider("Scene Change Threshold", 0.0, 100.0, 30.0)

rate_control = st.sidebar.radio("Rate Control", ["VBR (Constant QP)", "CBR (Target Bitrate)"])
if rate_control == "VBR (Constant QP)":
    target_qp = st.sidebar.slider("Target QP (Quality)", 1, 100, 20)
else:
    target_bitrate = st.sidebar.slider("Target Bits per Frame", 1000, 100000, 50000)
    initial_qp = st.sidebar.slider("Initial QP", 1, 100, 20)

quant_matrix_type = st.sidebar.selectbox("Quantization Matrix", ["flat", "weighted"])

motion_window = st.sidebar.slider("Motion Search Window (px)", 4, 32, 8)
block_size = st.sidebar.selectbox("Block Size", [8, 16], index=0)

# --- State Management ---
if 'processed_frames' not in st.session_state:
    st.session_state.processed_frames = []
if 'metrics' not in st.session_state:
    st.session_state.metrics = []

# --- File Upload ---
uploaded_file = st.sidebar.file_uploader("Upload a Video (MP4, AVI)", type=["mp4", "avi", "mov"])

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert to grayscale for simpler pedagogical visualization
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Resize to manageable size for real-time-ish processing
        gray = cv2.resize(gray, (256, 256))
        frames.append(gray)
    cap.release()
    return frames

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    with st.spinner("Loading video..."):
        original_frames = process_video(tfile.name)

    st.success(f"Loaded {len(original_frames)} frames.")

    if st.button("Start Compression"):
        st.session_state.processed_frames = []
        st.session_state.metrics = []

        num_frames = len(original_frames)

        if gop_auto:
            frame_types = ce.auto_gop_detect(original_frames, scene_threshold)
        else:
            frame_types = ce.parse_gop_structure(gop_structure, num_frames)

        ref_frame = None
        current_qp = target_qp if rate_control == "VBR (Constant QP)" else initial_qp

        progress_bar = st.progress(0)
        status_text = st.empty()

        col1, col2, col3 = st.columns(3)
        with col1:
            orig_placeholder = st.empty()
        with col2:
            recon_placeholder = st.empty()
        with col3:
            residual_placeholder = st.empty()

        metrics_placeholder = st.empty()

        for i in range(num_frames):
            frame = original_frames[i]
            ftype = frame_types[i] if codec_mode == "MPEG (Intra + Inter)" else 'I'

            h, w = frame.shape
            reconstructed_frame = np.zeros_like(frame)
            residual_frame = np.zeros_like(frame, dtype=np.float32)
            mvs = []

            frame_bits = 0

            # Process block by block
            for y in range(0, h, block_size):
                for x in range(0, w, block_size):
                    curr_block = frame[y:y+block_size, x:x+block_size]

                    if ftype == 'I' or ref_frame is None:
                        # Intra coding: just DCT and Quantize the block itself
                        pred_block = np.zeros_like(curr_block)
                        residual_block = curr_block.astype(np.float32)
                        mv = (0, 0)
                    else:
                        # Inter coding: Motion Estimation
                        mv, pred_block = ce.motion_search(curr_block, ref_frame, (y, x), motion_window)
                        residual_block = ce.compute_residual(curr_block, pred_block)
                        mvs.append(((x, y), mv))

                    # Transform & Quantize Residual
                    dct_block = ce.block_dct(residual_block)
                    q_block = ce.quantize(dct_block, current_qp, quant_matrix_type)

                    # Estimate bits for this block
                    frame_bits += ce.estimate_frame_bits(q_block)

                    # Dequantize & Inverse Transform
                    dq_block = ce.dequantize(q_block, current_qp, quant_matrix_type)
                    rec_residual = ce.block_idct(dq_block)

                    # Reconstruct
                    recon_block = ce.reconstruct_macroblock(pred_block, rec_residual)
                    reconstructed_frame[y:y+block_size, x:x+block_size] = recon_block
                    residual_frame[y:y+block_size, x:x+block_size] = np.abs(residual_block)

            # Rate Control update for next frame
            if rate_control == "CBR (Target Bitrate)":
                current_qp = ce.encode_cbr(target_bitrate, frame_bits, current_qp)

            psnr = ce.calculate_psnr(frame, reconstructed_frame)
            comp_ratio = (h * w * 8) / max(1, frame_bits)

            metrics = {
                "Frame": i,
                "Type": ftype,
                "QP": current_qp,
                "PSNR": psnr,
                "Bits": frame_bits,
                "Ratio": comp_ratio
            }

            st.session_state.processed_frames.append({
                "recon": reconstructed_frame,
                "resid": residual_frame,
                "mvs": mvs
            })
            st.session_state.metrics.append(metrics)

            # Update Visualizations
            orig_placeholder.image(frame, caption=f"Original Frame {i}", use_container_width=True)
            recon_placeholder.image(reconstructed_frame, caption=f"Reconstructed ({ftype})", use_container_width=True)

            # Normalize residual for visualization
            res_vis = cv2.normalize(residual_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            residual_placeholder.image(res_vis, caption="Residual (Error)", use_container_width=True)

            metrics_placeholder.metric("PSNR", f"{psnr:.2f} dB", delta=f"{ftype}")

            ref_frame = reconstructed_frame
            progress_bar.progress((i + 1) / num_frames)
            status_text.text(f"Processing frame {i+1}/{num_frames}...")

        st.success("Compression Complete!")

# --- Post-Processing Visualizations ---
if st.session_state.processed_frames:
    st.divider()
    st.header("Analysis & Visualization")

    frame_idx = st.slider("Select Frame to Analyze", 0, len(st.session_state.processed_frames)-1, 0)

    m = st.session_state.metrics[frame_idx]
    p = st.session_state.processed_frames[frame_idx]

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Frame Type", m["Type"])
    col_b.metric("PSNR", f"{m['PSNR']:.2f} dB")
    col_c.metric("Compression Ratio", f"{m['Ratio']:.1f}:1")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Reconstructed")
        st.image(p["recon"], use_container_width=True)

    with col2:
        st.subheader("Residual (Error)")
        # Normalize residual for better visibility
        res_norm = cv2.normalize(p["resid"], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        st.image(res_norm, use_container_width=True)
        st.caption("Brighter areas mean larger prediction errors.")

    with col3:
        st.subheader("Motion Vectors")
        mv_canvas = np.zeros((256, 256, 3), dtype=np.uint8) + 50
        for (x, y), (dy, dx) in p["mvs"]:
            # Draw arrow
            start_point = (x + block_size//2, y + block_size//2)
            end_point = (start_point[0] + dx, start_point[1] + dy)
            cv2.arrowedLine(mv_canvas, start_point, end_point, (0, 255, 0), 1, tipLength=0.5)
        st.image(mv_canvas, use_container_width=True)
        st.caption("Arrows show how blocks moved from the reference frame.")

    st.subheader("Metrics Over Time")
    psnrs = [m["PSNR"] for m in st.session_state.metrics]
    st.line_chart(psnrs)
    st.caption("PSNR per frame")
