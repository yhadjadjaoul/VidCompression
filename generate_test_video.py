import cv2
import numpy as np

def create_test_video(filename="test_video.mp4", width=256, height=256, fps=10, duration=2):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height), False)

    num_frames = fps * duration
    for i in range(num_frames):
        # Create a black frame
        frame = np.zeros((height, width), dtype=np.uint8)

        # Draw a moving white circle
        center_x = int(width / 2 + 50 * np.cos(2 * np.pi * i / num_frames))
        center_y = int(height / 2 + 50 * np.sin(2 * np.pi * i / num_frames))
        cv2.circle(frame, (center_x, center_y), 30, 255, -1)

        # Add some text to test intra-coding quality
        cv2.putText(frame, f"Frame {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 127, 2)

        out.write(frame)

    out.release()
    print(f"Created {filename}")

if __name__ == "__main__":
    create_test_video()
