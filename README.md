# VidCompression: Educational Video Codec

VidCompression is a pedagogical video compression tool designed to demonstrate the core principles of modern video codecs. Written entirely in Python and NumPy, it breaks down the complex process of video encoding into understandable steps, avoiding external black-box libraries for the compression logic itself.

## 🚀 Features

- **Transform Coding**: Implements 2D Discrete Cosine Transform (DCT) to transform spatial pixels into frequency coefficients.
- **Quantization**: Demonstrates lossy compression with adjustable Quality Parameters (QP) and support for custom quantization matrices (Flat vs. Weighted).
- **Motion Compensation**: Uses exhaustive search block-matching to perform inter-frame prediction, generating motion vectors and residuals.
- **Rate Control**: Implements basic Constant Bitrate (CBR) and Variable Bitrate (VBR) strategies.
- **GOP Management**: Supports manual Group of Pictures (GOP) patterns (e.g., `IPPP`) and automatic scene change detection for I-frame insertion.
- **Interactive Dashboard**: A Streamlit-based UI to visualize reconstructed frames, prediction residuals, and motion vector fields in real-time.

## 🛠️ Installation

Ensure you have Python installed, then install the required dependencies:

```bash
pip install numpy opencv-python-headless streamlit
```

## 📖 Usage

### 1. Generate a Test Video
If you don't have a small video file to test with, you can generate a synthetic one:
```bash
python generate_test_video.py
```
This will create a file named `test_video.mp4` in the current directory.

### 2. Launch the Application
Start the interactive dashboard using Streamlit:
```bash
streamlit run app.py
```

### 3. Compress and Analyze
- **Upload**: Use the sidebar to upload `test_video.mp4` or any other small video file.
- **Configure**: Adjust codec settings like Mode (MJPEG vs MPEG), Block Size, and Rate Control.
- **Run**: Click "Start Compression" to begin the encoding process.
- **Visualize**: After processing, use the "Analysis & Visualization" section to inspect individual frames, residuals, and motion vectors.

## 📂 Project Structure

- `app.py`: The Streamlit frontend and application logic.
- `codec_engine.py`: The core "brain" of the codec containing DCT, motion estimation, and quantization functions.
- `generate_test_video.py`: A utility script for creating simple test content.

## 🎓 How It Works

VidCompression follows the standard hybrid video coding model:
1. **Prediction**: For P-frames, the codec finds the best match for each block in the previous frame (Motion Estimation).
2. **Residual**: The difference between the actual block and the prediction is calculated.
3. **Transform**: The residual (or the block itself for I-frames) is converted to the frequency domain using DCT.
4. **Quantization**: High-frequency information is discarded based on the QP setting.
5. **Reconstruction**: The process is reversed to build the frame that a player would see, providing a loopback for future predictions.
