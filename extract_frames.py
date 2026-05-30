import cv2
import os
import argparse


def extract_frames(video_path, output_dir, prefix="frame"):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    frame_interval = max(1, int(round(fps)))
    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            second = frame_count // frame_interval
            filename = f"{prefix}_{second:04d}s.jpg"
            out_path = os.path.join(output_dir, filename)
            cv2.imwrite(out_path, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Done. Extracted {saved_count} frames from {frame_count} total frames "
          f"(FPS={fps:.2f}, interval={frame_interval}) → {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract one frame per second from a video"
    )
    parser.add_argument("video", help="Path to input video")
    parser.add_argument("-o", "--output", default="frames", help="Output directory (default: frames)")
    parser.add_argument("-p", "--prefix", default="frame", help="Filename prefix (default: frame)")
    args = parser.parse_args()

    extract_frames(args.video, args.output, args.prefix)
