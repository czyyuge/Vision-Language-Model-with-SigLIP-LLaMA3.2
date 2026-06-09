import argparse
import os
import shutil
from extract_frames import extract_frames
from batch_inference import batch_inference


def run(video_path, output_json, llama_path, checkpoint_path, frames_dir="frames", prompt=None, cleanup=True):
    # Step 1: extract frames
    print("=" * 50)
    print("Step 1/2: Extracting frames from video ...")
    print("=" * 50)
    extract_frames(video_path, frames_dir)

    # Step 2: batch inference
    print("\n" + "=" * 50)
    print("Step 2/2: Running VLM inference on frames ...")
    print("=" * 50)
    kwargs = {}
    if prompt:
        kwargs["prompt"] = prompt
    batch_inference(frames_dir, output_json, llama_path, checkpoint_path, **kwargs)

    # Step 3: cleanup temporary frames
    if cleanup and os.path.isdir(frames_dir):
        shutil.rmtree(frames_dir)
        print(f"\nCleaned up temporary frames directory: {frames_dir}")

    print("\nAll done. Results saved to", output_json)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Video → extract frames → VLM captions (one-click)"
    )
    parser.add_argument("video", help="Path to input video")
    parser.add_argument("-o", "--output", default="results.json", help="Output JSON path")
    parser.add_argument("--llama", required=True, help="Path to LLaMA-3.2-3B directory")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint .pth")
    parser.add_argument("--frames-dir", default="frames", help="Temporary frames directory")
    parser.add_argument("--prompt", default="Describe this image in detail.", help="Prompt for each frame")
    parser.add_argument("--keep-frames", action="store_true", help="Keep temporary frames directory after completion")
    args = parser.parse_args()

    run(args.video, args.output, args.llama, args.checkpoint, args.frames_dir, args.prompt, cleanup=not args.keep_frames)
